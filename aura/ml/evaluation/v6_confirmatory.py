"""Confirmatory analyses of ``docs/PREREGISTRATION_v6_2026-10-03.md`` (commit 61e777b).

    python -m aura.ml.evaluation.v6_confirmatory extract     # score untouched CheXpert patients
    python -m aura.ml.evaluation.v6_confirmatory simcheck    # torch simulator vs PennyLane
    python -m aura.ml.evaluation.v6_confirmatory c1          # primary: matched-optimiser arms
    python -m aura.ml.evaluation.v6_confirmatory c1b         # entanglement ablation, 10 seeds
    python -m aura.ml.evaluation.v6_confirmatory c2          # 36-config symmetric search
    python -m aura.ml.evaluation.v6_confirmatory summarize   # tests + reading -> v6_confirmatory.json

Every decision rule lives in the pre-registration. This file computes; it does not choose.
Development set = the existing ``chexpert_evidence.npz`` (first 8,000 patients). Test set =
every later patient under the same selection rule, scored here for the first time.
"""
from __future__ import annotations

import csv
import json
import sys
import time

import numpy as np
from scipy import stats

from aura.common.config import ARTIFACTS, ensure_dirs
from aura.common.mathx import softmax
from aura.services.safety.calibration import expected_calibration_error, fit_temperature

PART = ARTIFACTS / "v6"
DEV = ARTIFACTS / "chexpert_evidence.npz"
FRESH = PART / "chexpert_fresh_evidence.npz"
OUT = ARTIFACTS / "v6_confirmatory.json"
N_DEV_PATIENTS = 8000
SPLIT_SEED = 7
MARGIN = 0.010
N_BOOT = 10_000


# --------------------------------------------------------------------------- #
# Cohort: the same rule as chexpert_external.select_studies, without the cap
# --------------------------------------------------------------------------- #
def fresh_rows() -> list[dict]:
    from aura.ml.evaluation.chexpert_external import CHEXPERT_ROOT

    best: dict[str, tuple[int, dict]] = {}
    order: list[str] = []
    with open(CHEXPERT_ROOT / "train.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["Frontal/Lateral"] != "Frontal":
                continue
            parts = row["Path"].split("/")
            patient, study = parts[2], parts[3]
            if patient not in best:
                order.append(patient)
            idx = int(study.replace("study", ""))
            if patient not in best or idx > best[patient][0]:
                best[patient] = (idx, row)
    return [best[p][1] for p in order[N_DEV_PATIENTS:]]


def run_extract() -> None:
    """Score every untouched patient through the served seam, checkpointing every 2,000."""
    from PIL import Image
    from aura.ml.evaluation.chexpert_external import CHEXPERT_ROOT, map_diagnosis
    from aura.schemas.contracts import StructuredPriors
    from aura.services.fusion.evidence import encode
    from aura.services.vision.engine import VisionEngine

    rows = fresh_rows()
    print(f"[extract] {len(rows)} untouched patients", flush=True)
    ckpt = PART / "chexpert_fresh_partial.npz"
    X, y0, y1, keep = [], [], [], []
    start = 0
    if ckpt.exists():
        d = np.load(ckpt)
        X, y0, y1, keep = list(d["X"]), list(d["y_u0"]), list(d["y_u1"]), list(d["row"])
        start = int(d["next"])
        print(f"[extract] resuming at row {start}", flush=True)

    vision = VisionEngine.load()
    t0 = time.perf_counter()

    def save(nxt: int, path) -> None:
        np.savez(path, X=np.asarray(X, float), y_u0=np.asarray(y0, int),
                 y_u1=np.asarray(y1, int), row=np.asarray(keep, int), next=nxt)

    for i in range(start, len(rows)):
        row = rows[i]
        path = CHEXPERT_ROOT / row["Path"].split("CheXpert-v1.0-small/", 1)[-1]
        if path.is_file():
            try:
                img = np.array(Image.open(path).convert("L"), dtype=float)
            except OSError:
                img = None
            if img is not None:
                vr = vision.analyze(f"chexpert-fresh-{i}", img)
                X.append(encode(vr, StructuredPriors()))
                y0.append(map_diagnosis(row, 0))
                y1.append(map_diagnosis(row, 1))
                keep.append(i)
        if (i + 1) % 2000 == 0:
            save(i + 1, ckpt)
            rate = (i + 1 - start) / (time.perf_counter() - t0)
            print(f"  [{i + 1}/{len(rows)}] {rate:.2f} img/s", flush=True)
    save(len(rows), FRESH)
    print(f"[extract] wrote {FRESH} n={len(X)} (of {len(rows)} rows)", flush=True)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def dev_splits(labels: str = "y_u0"):
    d = np.load(DEV)
    X, y = d["X"], d[labels]
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(X))
    n_tr = int(0.8 * len(X))
    tr, cal = perm[:n_tr], perm[n_tr:]
    return X[tr], y[tr], X[cal], y[cal]


def test_set(labels: str = "y_u0"):
    d = np.load(FRESH)
    return d["X"], d[labels]


# --------------------------------------------------------------------------- #
# Fast, verified circuit simulation
# --------------------------------------------------------------------------- #
def torch_circuit(n_qubits: int, n_layers: int, entangler: str):
    """Batched statevector of the served fusion circuit (``device.make_qnode``):
    RY(pi*x_i) on qubit i; per layer RY(theta0), RZ(theta1) per qubit, then the entangler;
    readout <Z_i>. Returns f(X[B,n], theta[L,n,2]) -> [B,n]. Used only if ``simcheck``
    passes (pre-registration §2)."""
    import torch

    from aura.services.fusion.device import entangler_pairs

    pairs = entangler_pairs(entangler, n_qubits)
    n = n_qubits

    def apply1(psi, q, m):            # psi [B, 2,...,2]; m [2,2] or [B,2,2]
        psi = psi.movedim(q + 1, -1)
        if m.dim() == 2:
            psi = psi @ m.T
        else:
            shp = psi.shape
            psi = torch.einsum("bij,b...j->b...i", m, psi.reshape(shp[0], -1, 2)).reshape(shp)
        return psi.movedim(-1, q + 1)

    def cnot(psi, c, t):
        psi = psi.clone()
        idx1 = [slice(None)] * (n + 1)
        idx1[c + 1] = 1
        sub = psi[tuple(idx1)]
        tt = t if t < c else t - 1
        psi[tuple(idx1)] = sub.flip(tt + 1)
        return psi

    def ry(th):
        c, s = torch.cos(th / 2), torch.sin(th / 2)
        return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)

    def rz(th):
        e = torch.exp(-0.5j * th.to(torch.complex128))
        return torch.stack([torch.stack([e, torch.zeros_like(e)], -1),
                            torch.stack([torch.zeros_like(e), e.conj()], -1)], -2)

    def f(X, theta):
        X = X.to(torch.float64)
        B = X.shape[0]
        psi = torch.zeros((B,) + (2,) * n, dtype=torch.complex128)
        psi[(slice(None),) + (0,) * n] = 1.0
        for i in range(n):
            psi = apply1(psi, i, ry(np.pi * X[:, i]).to(torch.complex128))
        for layer in range(n_layers):
            for i in range(n):
                psi = apply1(psi, i, ry(theta[layer, i, 0]).to(torch.complex128))
                psi = apply1(psi, i, rz(theta[layer, i, 1]))
            for c, t in pairs:
                psi = cnot(psi, c, t)
        p = (psi.real ** 2 + psi.imag ** 2)
        out = []
        for i in range(n):
            m = p.movedim(i + 1, -1).reshape(B, -1, 2).sum(1)
            out.append(m[:, 0] - m[:, 1])
        return torch.stack(out, -1)

    return f


def run_simcheck() -> dict:
    import torch

    from aura.services.fusion.device import make_qnode

    rng = np.random.default_rng(0)
    rows = []
    for nq in (7, 8):
        for ent in ("ring", "none"):
            L = 3
            qn = make_qnode(nq, L, interface="torch", entangler=ent)
            fast = torch_circuit(nq, L, ent)
            worst = 0.0
            for _ in range(10):                      # 10 batches x 100 = 1,000 pairs
                X = torch.tensor(rng.uniform(0, 1, (100, nq)))
                th = torch.tensor(rng.normal(0, 1.0, (L, nq, 2)))
                ref = torch.stack([z.to(torch.float64) for z in qn(X, th)], -1)
                worst = max(worst, float((ref - fast(X, th)).abs().max()))
            rows.append({"qubits": nq, "entangler": ent, "pairs": 1000, "max_abs_diff": worst})
            print(rows[-1], flush=True)
    res = {"criterion": "max |diff| < 1e-8", "checks": rows,
           "passed": all(r["max_abs_diff"] < 1e-8 for r in rows)}
    (PART / "simcheck.json").write_text(json.dumps(res, indent=2))
    return res


def _use_fast() -> bool:
    p = PART / "simcheck.json"
    return p.exists() and json.loads(p.read_text())["passed"]


def _build_quantum_fast(n_qubits=8, n_layers=3, entangler="ring", scale=1.0, init_sd=0.1):
    import torch

    from aura.ml.evaluation.matched_optimiser import N_DX

    circ = torch_circuit(n_qubits, n_layers, entangler)

    def build():
        theta = torch.nn.Parameter(init_sd * torch.randn(n_layers, n_qubits, 2,
                                                         dtype=torch.float64))
        W = torch.nn.Parameter(0.1 * torch.randn(N_DX, n_qubits, dtype=torch.float64))
        b = torch.nn.Parameter(torch.zeros(N_DX, dtype=torch.float64))
        return [theta, W, b], lambda X: circ(scale * X[:, :n_qubits], theta) @ W.T + b
    return build


# --------------------------------------------------------------------------- #
# C1 — matched-optimiser comparison, dev -> untouched test
# --------------------------------------------------------------------------- #
def _save_arms(tag: str, arms: dict, extra: dict, yte: np.ndarray) -> None:
    np.savez(PART / f"{tag}_probs.npz", yte=yte,
             **{f"probs_{k}": v["_probs"] for k, v in arms.items()})
    clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
             for k, v in arms.items()}
    (PART / f"{tag}.json").write_text(json.dumps({"arms": clean, **extra}, indent=2,
                                                 default=float))


def _save_params(path, params) -> None:
    np.savez(path, **{f"p{i}": p.detach().numpy() for i, p in enumerate(params)})


def _load_forward(build, path):
    """Rebuild an arm and load its saved parameters; returns the forward function."""
    import torch

    params, forward = build()
    z = np.load(path)
    with torch.no_grad():
        for i, p in enumerate(params):
            p.copy_(torch.tensor(z[f"p{i}"]))
    return forward


def _logits(forward, X):
    import torch

    with torch.no_grad():
        return forward(torch.tensor(X, dtype=torch.float64)).numpy()


def _c1_builders(d):
    import aura.ml.evaluation.matched_optimiser as mo

    q = _build_quantum_fast() if _use_fast() else mo._build_quantum()
    return {"classical": mo._build_linear(d), "learnable": mo._build_gated(d), "quantum": q}


def run_c1(labels: str = "y_u0") -> None:
    """Train on the development set only (no test data touched); parameters saved.
    Same trainer, grid and selection rule as ``matched_optimiser._search``."""
    import aura.ml.evaluation.matched_optimiser as mo

    Xtr, ytr, Xcal, ycal = dev_splits(labels)
    fitting = {}
    for name, build in _c1_builders(Xtr.shape[1]).items():
        best, runs = None, []
        path = PART / f"C1_{labels}_{name}_params.npz"
        for lr in mo.LR_GRID:
            t0 = time.time()
            rec, params, _ = mo._train_under_protocol(build, Xtr, ytr, Xcal, ycal, lr)
            rec["seconds"] = round(time.time() - t0, 1)
            runs.append({k: v for k, v in rec.items() if k != "history"})
            print(f"[C1:{labels}] {name} lr={lr} cal_nll={rec['best_calibration_nll']:.4f} "
                  f"epochs={rec['epochs_run']} ({rec['seconds']}s)", flush=True)
            if best is None or rec["best_calibration_nll"] < best["best_calibration_nll"]:
                best = rec
                _save_params(path, params)
                n_params = int(sum(p.numel() for p in params))
        fitting[name] = {"selected_lr": best["lr"], "best_epoch": best["best_epoch"],
                         "epochs_run": best["epochs_run"],
                         "converged_by_early_stopping": best["stopped_early"],
                         "parameters": n_params, "grid": runs}
    (PART / f"C1_{labels}_fitting.json").write_text(json.dumps(
        {"fitting": fitting, "simulator": "torch" if _use_fast() else "pennylane",
         "n_train": len(ytr), "n_cal": len(ycal)}, indent=2, default=float))


def score_c1(labels: str = "y_u0") -> None:
    """Score the frozen C1 arms once on the untouched test set."""
    import aura.ml.evaluation.matched_optimiser as mo

    Xtr, ytr, Xcal, ycal = dev_splits(labels)
    Xte, yte = test_set(labels)
    meta = json.loads((PART / f"C1_{labels}_fitting.json").read_text())
    arms = {}
    for name, build in _c1_builders(Xtr.shape[1]).items():
        fwd = _load_forward(build, PART / f"C1_{labels}_{name}_params.npz")
        arms[name] = mo._evaluate(_logits(fwd, Xcal), _logits(fwd, Xte), ycal, yte)
    Lcal, Lte, p, anchor = mo._lbfgs_anchor(Xtr, ytr, Xcal, ycal, Xte)
    arms["logistic"] = mo._evaluate(Lcal, Lte, ycal, yte)
    meta["fitting"]["logistic"] = {**anchor, "parameters": p}
    _save_arms(f"C1_{labels}", arms, {**meta, "n_test": len(yte)}, yte)


# --------------------------------------------------------------------------- #
# C1b — entanglement ablation (addendum (a) analysis-B protocol)
# --------------------------------------------------------------------------- #
def run_c1b() -> None:
    """Addendum (a) analysis-B protocol, trained on the development split only.
    ``train_quantum`` is the published trainer; parameters are saved and scored later."""
    from aura.ml.evaluation import revision_v51 as rv
    from aura.ml.training import train_fusion
    from aura.ml.training.train_fusion import train_quantum

    if _use_fast():                     # verified-identical simulator (prereg §2)
        train_fusion.make_qnode = _fast_qnode
    Xtr, ytr, _, _ = dev_splits()
    Xtr = Xtr[:, :rv.N_QUBITS]
    for seed in rv.SEEDS:
        for ent in ("ring", "none"):
            path = PART / f"C1b_seed{seed}_{ent}_params.npz"
            if path.exists():
                continue
            t0 = time.time()
            th, W, b = train_quantum(Xtr, ytr, rv.N_QUBITS, rv.N_LAYERS, epochs=rv.EPOCHS,
                                     seed=seed, entangler=ent)
            np.savez(path, theta=np.asarray(th), W=np.asarray(W), b=np.asarray(b))
            print(f"[C1b] seed {seed} {ent} ({time.time() - t0:.0f}s)", flush=True)


def _fast_qnode(n_qubits, n_layers, device_name="default.qubit", interface="torch",
                entangler="ring", **_):
    """Drop-in for ``device.make_qnode`` returning the list of <Z_i> like the qnode."""
    import torch

    f = torch_circuit(n_qubits, n_layers, entangler)

    def circuit(X, theta):
        out = f(torch.as_tensor(X), torch.as_tensor(theta))
        return list(out.unbind(-1)) if interface == "torch" else list(out.numpy().T)
    return circuit


def _quantum_logits(theta, W, b, X, n_qubits, n_layers, entangler):
    z = np.stack(_fast_qnode(n_qubits, n_layers, interface="numpy",
                             entangler=entangler)(X, theta), axis=-1)
    return z @ W.T + b


def score_c1b() -> None:
    from aura.ml.evaluation import revision_v51 as rv

    _, _, Xcal, ycal = dev_splits()
    Xte, yte = test_set()
    Xcal, Xte = Xcal[:, :rv.N_QUBITS], Xte[:, :rv.N_QUBITS]
    rows = []
    for seed in rv.SEEDS:
        res = {}
        for ent in ("ring", "none"):
            z = np.load(PART / f"C1b_seed{seed}_{ent}_params.npz")

            def lg(X, z=z, ent=ent):
                return _quantum_logits(z["theta"], z["W"], z["b"], X, rv.N_QUBITS,
                                       rv.N_LAYERS, ent)
            res[ent] = rv._calibrated(lg(Xcal), ycal, lg(Xte), yte)
        np.savez(PART / f"C1b_seed{seed}.npz", ring=res["ring"][1], none=res["none"][1])
        rows.append({"seed": seed, "entangled": res["ring"][0], "product": res["none"][0],
                     "delta_nll": round(res["ring"][0]["nll"] - res["none"][0]["nll"], 5)})
        print(rows[-1], flush=True)
    (PART / "C1b_partial.json").write_text(json.dumps(rows, indent=2))


# --------------------------------------------------------------------------- #
# C2 — 36-configuration symmetric search per arm
# --------------------------------------------------------------------------- #
Q_SPACE = [(L, s, sd, lr) for L in (1, 2, 3, 4, 6) for s in (0.5, 1.0, 2.0)
           for sd in (0.01, 0.1, 1.0) for lr in (0.01, 0.05, 0.1)]
C_SPACE = [(lr, wd, sd) for lr in (0.003, 0.01, 0.05, 0.1, 0.3)
           for wd in (0.0, 1e-5, 1e-4, 1e-3) for sd in (0.01, 0.1, 1.0)]
BUDGET = 36


def _draw(space):
    idx = np.random.default_rng(SPLIT_SEED).choice(len(space), BUDGET, replace=False)
    return [space[i] for i in sorted(idx)]


def _build_linear_c(d, sd):
    import torch
    from aura.ml.evaluation.matched_optimiser import N_DX

    def build():
        W = torch.nn.Parameter(sd * torch.randn(N_DX, d, dtype=torch.float64))
        b = torch.nn.Parameter(torch.zeros(N_DX, dtype=torch.float64))
        return [W, b], lambda X: X @ W.T + b
    return build


def _build_gated_c(d, sd):
    import torch
    from aura.ml.evaluation.matched_optimiser import N_DX

    def build():
        # the published init uses 0.05 for A and 0.1 for W; scale both by sd/0.1
        A = torch.nn.Parameter(0.5 * sd * torch.randn(d, d, dtype=torch.float64))
        c = torch.nn.Parameter(torch.zeros(d, dtype=torch.float64))
        W = torch.nn.Parameter(sd * torch.randn(N_DX, d, dtype=torch.float64))
        b = torch.nn.Parameter(torch.zeros(N_DX, dtype=torch.float64))
        return [A, c, W, b], lambda X: (torch.sigmoid(X @ A.T + c) * X) @ W.T + b
    return build


def run_c2(arm: str) -> None:
    import aura.ml.evaluation.matched_optimiser as mo

    Xtr, ytr, Xcal, ycal = dev_splits()
    d = Xtr.shape[1]
    log_path = PART / f"C2_{arm}_grid.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else []
    done = {json.dumps(r["config"]) for r in log}
    configs = _draw(Q_SPACE if arm == "quantum" else C_SPACE)
    for cfg in configs:
        key = json.dumps(list(cfg))
        if key in done:
            continue
        build = _c2_build(arm, cfg, d)
        lr = cfg[3] if arm == "quantum" else cfg[0]
        wd = 0.0 if arm == "quantum" else cfg[1]
        mo.WEIGHT_DECAY = wd
        t0 = time.time()
        rec, params, _ = mo._train_under_protocol(build, Xtr, ytr, Xcal, ycal, lr)
        _save_params(PART / f"C2_{arm}_{len(log):02d}.npz", params)
        log.append({"config": list(cfg), "file": f"C2_{arm}_{len(log):02d}.npz",
                    "best_calibration_nll": rec["best_calibration_nll"],
                    "best_epoch": rec["best_epoch"], "epochs_run": rec["epochs_run"],
                    "parameters": int(sum(p.numel() for p in params)),
                    "seconds": round(time.time() - t0, 1)})
        log_path.write_text(json.dumps(log, indent=2))
        print(f"[C2:{arm}] {cfg} cal_nll={rec['best_calibration_nll']:.4f} "
              f"({log[-1]['seconds']}s)", flush=True)
    mo.WEIGHT_DECAY = 0.0


# --------------------------------------------------------------------------- #
# Summaries and the pre-registered reading
# --------------------------------------------------------------------------- #
def _pnll(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))


def reading(d: np.ndarray) -> dict:
    from aura.ml.evaluation.revision_v51 import tost

    t = tost(d, MARGIN)
    m, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    lo95, hi95 = stats.t.interval(0.95, len(d) - 1, loc=m, scale=se)
    lo90, hi90 = t["ci90"]
    inside = lo90 > -MARGIN and hi90 < MARGIN
    if inside and lo95 <= 0 <= hi95:
        verdict = "equivalent within ±0.01 nats"
    elif inside:
        verdict = ("detectable difference, quantum " + ("worse" if m > 0 else "better")
                   + ", smaller than the margin; practically equivalent")
    elif lo95 > MARGIN:
        verdict = "quantum materially worse"
    elif lo95 > 0:
        verdict = "quantum worse; size relative to margin unresolved"
    elif hi95 < 0:
        verdict = "A QUANTUM BENEFIT SURVIVED"
    else:
        verdict = "inconclusive"
    return {"contrast": "NLL_quantum - NLL_poe (positive = quantum worse)",
            "tost": t, "ci95": [round(float(lo95), 5), round(float(hi95), 5)],
            "p_two_sided": float(f"{2 * stats.t.sf(abs(m / se), len(d) - 1):.3e}"),
            "reading": verdict}


def _boot_ci(fn, n, seed=0, B=2000):
    rng = np.random.default_rng(seed)
    v = np.array([fn(rng.integers(0, n, n)) for _ in range(B)])
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def _c1_block(tag: str) -> dict:
    from aura.ml.evaluation.matched_optimiser import _holm, _mcnemar

    meta = json.loads((PART / f"{tag}.json").read_text())
    d = np.load(PART / f"{tag}_probs.npz")
    y = d["yte"]
    names = ["classical", "quantum", "learnable", "logistic"]
    P = {k: d[f"probs_{k}"] for k in names}
    for k in names:
        meta["arms"][k]["ece_ci95_boot"] = _boot_ci(
            lambda i, k=k: expected_calibration_error(P[k][i], y[i]), len(y))
    pairwise, ap, np_ = {}, {}, {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            key = f"{a}_vs_{b}"
            ca, cb = P[a].argmax(1) == y, P[b].argmax(1) == y
            diff = _pnll(P[a], y) - _pnll(P[b], y)
            w = float(stats.wilcoxon(diff).pvalue)
            mc = _mcnemar(ca, cb)
            pairwise[key] = {"delta_accuracy": round(float(ca.mean() - cb.mean()), 5),
                             "mcnemar": mc, "delta_nll": round(float(diff.mean()), 5),
                             "nll_wilcoxon_p": float(f"{w:.3e}")}
            ap[key], np_[key] = mc["p"], w
    for k, v in _holm(ap).items():
        pairwise[k]["mcnemar_p_holm"] = v
    for k, v in _holm(np_).items():
        pairwise[k]["nll_wilcoxon_p_holm"] = float(f"{v:.3e}")
    meta["pairwise"] = pairwise
    meta["primary"] = reading(_pnll(P["quantum"], y) - _pnll(P["classical"], y))
    counts = np.bincount(y, minlength=P["quantum"].shape[1])
    meta["majority_class_floor"] = round(float(counts.max() / len(y)), 4)
    return meta


def _c1b_block() -> dict:
    from aura.ml.evaluation import revision_v51 as rv

    rows = json.loads((PART / "C1b_partial.json").read_text())
    y = test_set()[1]
    ring = np.array([np.load(PART / f"C1b_seed{r['seed']}.npz")["ring"] for r in rows])
    none = np.array([np.load(PART / f"C1b_seed{r['seed']}.npz")["none"] for r in rows])
    per_patient = np.mean([_pnll(ring[s], y) - _pnll(none[s], y) for s in range(len(rows))], 0)
    rng = np.random.default_rng(0)
    n = len(y)
    boot = np.array([per_patient[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    verdict = ("the entanglement penalty replicates on untouched patients" if lo > 0 else
               "reversal" if hi < 0 else "not replicated on untouched patients")
    return {"per_seed": rows, "seed_level": rv._seed_stats([r["delta_nll"] for r in rows]),
            "patient_bootstrap": {"point": round(float(per_patient.mean()), 5),
                                  "ci95": [round(float(lo), 5), round(float(hi), 5)],
                                  "n_resamples": N_BOOT},
            "tost": rv.tost(per_patient, MARGIN), "reading": verdict}


def _c2_build(arm, cfg, d):
    if arm == "quantum":
        L, sc, sd, _ = cfg
        return _build_quantum_fast(8, L, "ring", sc, sd)
    _, _, sd = cfg
    return (_build_linear_c if arm == "classical" else _build_gated_c)(d, sd)


def _c2_block() -> dict:
    Xtr, ytr, Xcal, ycal = dev_splits()
    _, yte = test_set()
    out, probs = {}, {}
    for arm in ("quantum", "classical", "learnable"):
        log = json.loads((PART / f"C2_{arm}_grid.json").read_text())
        best = min(log, key=lambda r: r["best_calibration_nll"])
        fwd = _load_forward(_c2_build(arm, best["config"], Xtr.shape[1]), PART / best["file"])
        Lcal, Lte = _logits(fwd, Xcal), _logits(fwd, test_set()[0])
        T = fit_temperature(Lcal, ycal)
        P = np.array([softmax(r / T) for r in Lte])
        probs[arm] = P
        edge = None
        if arm == "quantum":
            L, s, sd, _ = best["config"]
            edge = L == 6 or s in (0.5, 2.0) or sd == 1.0
        out[arm] = {"configs_run": len(log), "selected": best, "temperature": round(float(T), 4),
                    "nll": round(float(_pnll(P, yte).mean()), 5),
                    "accuracy": round(float((P.argmax(1) == yte).mean()), 5),
                    "ece": round(float(expected_calibration_error(P, yte)), 5),
                    **({"search_boundary_reached": bool(edge)} if edge is not None else {})}
    out["primary"] = reading(_pnll(probs["quantum"], yte) - _pnll(probs["classical"], yte))
    out["quantum_vs_gated"] = reading(_pnll(probs["quantum"], yte) - _pnll(probs["learnable"], yte))
    return out


def summarize() -> None:
    res = {"preregistration": "docs/PREREGISTRATION_v6_2026-10-03.md (commit 61e777b)",
           "development": {"source": "chexpert_evidence.npz (first 8,000 patients)",
                           "split": "80/20 train/calibration, default_rng(7)"},
           "test": {"source": "every later CheXpert patient, same selection rule",
                    "n": int(len(test_set()[1]))}}
    sc = PART / "simcheck.json"
    if sc.exists():
        res["simulator_check"] = json.loads(sc.read_text())
    for lab in ("y_u0", "y_u1"):
        if (PART / f"C1_{lab}_fitting.json").exists():
            score_c1(lab)
    if (PART / "C1b_seed41_none_params.npz").exists():
        score_c1b()
    for tag, key in (("C1_y_u0", "C1"), ("C1_y_u1", "C1_uncertain_as_positive")):
        if (PART / f"{tag}.json").exists():
            res[key] = _c1_block(tag)
    if (PART / "C1b_partial.json").exists():
        res["C1b"] = _c1b_block()
    if all((PART / f"C2_{a}_grid.json").exists() for a in ("quantum", "classical", "learnable")):
        res["C2"] = _c2_block()
    OUT.write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {OUT}")
    for k in ("C1", "C1_uncertain_as_positive", "C2"):
        if k in res:
            print(k, res[k]["primary"]["reading"], res[k]["primary"]["tost"]["ci90"])
    if "C1b" in res:
        print("C1b", res["C1b"]["reading"], res["C1b"]["patient_bootstrap"])


if __name__ == "__main__":
    ensure_dirs()
    PART.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "extract":
        run_extract()
    elif cmd == "simcheck":
        print(run_simcheck())
    elif cmd == "c1":
        run_c1(sys.argv[2] if len(sys.argv) > 2 else "y_u0")
    elif cmd == "c1b":
        run_c1b()
    elif cmd == "c2":
        run_c2(sys.argv[2])
    elif cmd == "summarize":
        summarize()
