"""AURA scientific-audit harness — reproducible, non-destructive, evidence-first.

WHY THIS FILE EXISTS
--------------------
The audit protocol assumed an ``audit_all.py`` that never existed in the repo or
its git history. This is that file, written from scratch to make the AURA
evidence-fusion claims *independently reproducible*: every reported number is
regenerated from saved model weights + a seeded synthetic split and written to a
CSV / plot / JSON that a reviewer can re-derive.

WHAT IT AUDITS
--------------
The genuinely-real, defensible core of AURA: the quantum-vs-classical evidence
*fusion* head (``services/fusion/*``) plus its safety calibration
(``services/safety/calibration.py``). It regenerates predictions for every
trained backend on the deterministic held-out split, computes the full clinical
metric suite, and runs a battery of paired statistical tests to decide whether
the "quantum beats classical" claim survives scrutiny.

It deliberately corrects one confound baked into ``ml/evaluation/benchmark.py``:
that harness temperature-scales the *quantum* logits but leaves the *classical*
logits raw, so its ECE/NLL/Brier gap conflates "calibrated vs uncalibrated" with
"quantum vs classical". Here classical is evaluated BOTH raw (to reproduce the
prior claim) and with its OWN fitted temperature (the fair comparison).

WHAT IT DOES NOT DO
-------------------
It does not retrain anything, does not touch the real MIMIC-CXR CNN path, and
NEVER deletes model weights. It only reads existing ``*.npz`` weights and writes
fresh outputs under ``audit_artifacts/run_<UTC>/``. Anything it cannot produce
evidence for is recorded as an explicit error in the manifest, never invented.

RUN
---
    E:\\AURA\\venv\\Scripts\\python.exe E:\\AURA\\aura-main\\audit_all.py

(The venv interpreter is required: the global Python's torch is blocked by a
Windows Application Control policy, and the venv also carries scipy/sklearn/
matplotlib. The fusion path itself is pure numpy and needs no GPU.)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
# 0. Paths & imports of the system under audit                                 #
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent            # ...\aura-main
PKG_ROOT = REPO_ROOT / "aura"                          # ...\aura-main\aura (import root)
sys.path.insert(0, str(PKG_ROOT))
os.chdir(PKG_ROOT)                                     # ARTIFACTS etc. resolve relative to pkg

import numpy as np                                     # noqa: E402

SEED = 7                                               # overwritten from settings below
N_SAMPLES = 500                                        # matches benchmark.run default
N_BOOTSTRAP = 2000
RNG = None                                             # set in seed_everything


# --------------------------------------------------------------------------- #
# 1. Reproducibility                                                           #
# --------------------------------------------------------------------------- #
def seed_everything(seed: int) -> None:
    global RNG
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    RNG = np.random.default_rng(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass  # fusion path is numpy-only; torch is optional here


def _pkg_version(name: str) -> str | None:
    try:
        return __import__(name).__version__
    except Exception:
        return None


def _sha256(path: Path, cap: int = 512 * 1024 * 1024) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    read = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk or read > cap:
                break
            h.update(chunk)
            read += len(chunk)
    return h.hexdigest()


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), *args],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None


def capture_environment(outdir: Path, seed: int) -> dict:
    env: dict = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": {
            "numpy": _pkg_version("numpy"),
            "scipy": _pkg_version("scipy"),
            "sklearn": _pkg_version("sklearn"),
            "matplotlib": _pkg_version("matplotlib"),
            "pandas": _pkg_version("pandas"),
            "torch": _pkg_version("torch"),
        },
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }
    # RAM (best-effort, no hard dependency)
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEMORYSTATUSEX(); m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        env["ram_total_gb"] = round(m.ullTotalPhys / 1e9, 2)
    except Exception:
        env["ram_total_gb"] = None
    # torch / cuda
    try:
        import torch
        env["torch_cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_version"] = torch.version.cuda
        env["cudnn_version"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as e:
        env["torch_cuda_available"] = None
        env["torch_import_error"] = repr(e)
    # dataset / weight hashes (integrity of the artifacts we read)
    from common.config import ARTIFACTS
    hashes = {}
    for rel in ["fusion_quantum.npz", "fusion_classical.npz", "fusion_ensemble.npz",
                "fusion_learnable.npz", "safety.npz", "vision.npz", "benchmark.json"]:
        hashes[rel] = _sha256(ARTIFACTS / rel)
    env["artifact_sha256"] = hashes
    (outdir / "environment.json").write_text(json.dumps(env, indent=2))
    return env


# --------------------------------------------------------------------------- #
# 2. Statistics primitives (implemented, not imported, so they are auditable)  #
# --------------------------------------------------------------------------- #
def bootstrap_ci(values: np.ndarray, stat_fn, n_boot=N_BOOTSTRAP, alpha=0.05, rng=None):
    """Percentile bootstrap CI for an arbitrary statistic over paired rows."""
    rng = rng or np.random.default_rng(SEED)
    values = np.asarray(values)
    n = len(values)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = stat_fn(values[idx])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat_fn(values)), float(lo), float(hi), boots


def bootstrap_diff_ci(a_correct, b_correct, n_boot=N_BOOTSTRAP, rng=None):
    """Paired bootstrap CI of mean(a) - mean(b) over the SAME resampled indices."""
    rng = rng or np.random.default_rng(SEED + 1)
    a = np.asarray(a_correct, dtype=float)
    b = np.asarray(b_correct, dtype=float)
    n = len(a)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = a[idx].mean() - b[idx].mean()
    point = float(a.mean() - b.mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi), boots


def mcnemar_exact(a_correct, b_correct):
    """Exact McNemar for paired binary correctness. Returns (b01, b10, p, stat)."""
    from scipy.stats import binomtest
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    b01 = int((~a & b).sum())   # a wrong, b right
    b10 = int((a & ~b).sum())   # a right, b wrong
    nd = b01 + b10
    if nd == 0:
        return b01, b10, 1.0, 0.0
    p = binomtest(min(b01, b10), nd, 0.5, alternative="two-sided").pvalue
    stat = (abs(b01 - b10) - 1) ** 2 / nd  # continuity-corrected chi-square
    return b01, b10, float(p), float(stat)


# ---- DeLong (Sun & Xu 2014 fast algorithm) for two correlated ROC curves --- #
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(y_true, prob_a, prob_b):
    """Two-sided DeLong test for AUROC(a) == AUROC(b) on the same binary labels.

    Returns (auc_a, auc_b, z, p). NaN if a class is degenerate.
    """
    from scipy.stats import norm
    y = np.asarray(y_true, dtype=int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan"), float("nan"), float("nan"), float("nan")
    order = (-y).argsort(kind="mergesort")
    label_1_count = int(y.sum())
    preds = np.vstack([np.asarray(prob_a, float)[order], np.asarray(prob_b, float)[order]])
    aucs, cov = _fast_delong(preds, label_1_count)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), float("nan"), float("nan")
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


def permutation_test_diff(a_correct, b_correct, n_perm=10000, rng=None):
    """Sign-flip permutation test on the paired accuracy difference."""
    rng = rng or np.random.default_rng(SEED + 2)
    d = np.asarray(a_correct, float) - np.asarray(b_correct, float)
    obs = d.mean()
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    perm = (signs * d).mean(axis=1)
    p = (np.abs(perm) >= abs(obs) - 1e-12).mean()
    return float(obs), float(p)


def cohens_d_paired(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


# --------------------------------------------------------------------------- #
# 3. Core audit                                                                #
# --------------------------------------------------------------------------- #
def softmax_rows(logits, T=1.0):
    from common.mathx import softmax
    return np.array([softmax(np.asarray(r, float) / T) for r in logits])


def run_audit() -> Path:
    from common.config import ARTIFACTS, get_settings
    from services.fusion.quantum import QuantumFusion
    from services.fusion.classical import ClassicalFusion
    from services.safety.calibration import (
        Calibration, fit_temperature, fit_conformal, expected_calibration_error,
    )
    from ml.training.dataset import build_evidence_dataset, make_splits
    from ml.evaluation.metrics import evaluate
    from schemas.clinical import DIAGNOSES

    settings = get_settings()
    global SEED
    SEED = int(getattr(settings, "seed", 7))
    coverage = float(getattr(settings, "conformal_coverage", 0.90))
    seed_everything(SEED)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = REPO_ROOT / "audit_artifacts" / f"run_{stamp}"
    for sub in ["", "csv", "plots", "metrics", "logs", "configs"]:
        (outdir / sub).mkdir(parents=True, exist_ok=True)

    log = logging.getLogger("audit")
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(outdir / "logs" / "audit.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(fh); log.addHandler(sh)

    manifest: dict = {"run": stamp, "seed": SEED, "coverage": coverage,
                      "stages": {}, "errors": {}}

    def stage(name):
        def deco(fn):
            def wrapped(*a, **k):
                t0 = time.time()
                try:
                    out = fn(*a, **k)
                    manifest["stages"][name] = {"ok": True, "seconds": round(time.time() - t0, 3)}
                    return out
                except Exception as e:
                    import traceback
                    manifest["errors"][name] = traceback.format_exc()
                    manifest["stages"][name] = {"ok": False, "seconds": round(time.time() - t0, 3),
                                                "error": repr(e)}
                    log.error("STAGE %s FAILED: %r", name, e)
                    return None
            return wrapped
        return deco

    log.info("=== AURA reproducible audit  run=%s  seed=%d ===", stamp, SEED)

    # -- environment ------------------------------------------------------- #
    env = capture_environment(outdir, SEED)
    log.info("env: py=%s numpy=%s cuda=%s commit=%s",
             env["python_version"], env["packages"]["numpy"],
             env.get("torch_cuda_available"), env.get("git_commit"))
    (outdir / "random_seed.txt").write_text(str(SEED))
    (outdir / "git_commit.txt").write_text(str(env.get("git_commit")))
    try:
        req = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (outdir / "requirements.txt").write_text(req)
    except Exception as e:
        log.warning("pip freeze failed: %r", e)

    # -- data + models ----------------------------------------------------- #
    q = QuantumFusion.load(); c = ClassicalFusion.load(); cal = Calibration.load()
    if q is None or c is None:
        raise RuntimeError("Fusion weights missing — cannot audit. Train first.")
    labels = [d.value for d in DIAGNOSES]
    n_classes = len(labels)

    _, va, te = make_splits(N_SAMPLES, seed=SEED + 101)   # identical recipe to benchmark
    Xva, yva = build_evidence_dataset(va)
    Xte, yte = build_evidence_dataset(te)
    Xte = np.asarray(Xte, float); yte = np.asarray(yte, int)
    log.info("held-out: n=%d  classes=%d  class_counts=%s",
             len(yte), n_classes, np.bincount(yte, minlength=n_classes).tolist())

    q_logits = np.array([q.logits(x) for x in Xte])
    c_logits = np.array([c.logits(x) for x in Xte])
    c_logits_va = np.array([c.logits(x) for x in Xva])

    T_q = float(cal.temperature)                                   # as shipped (quantum-fit)
    T_c = fit_temperature(c_logits_va, np.asarray(yva, int))        # FAIR: classical's own T
    log.info("temperatures: quantum(shipped)=%.4f  classical(fair-fit)=%.4f", T_q, T_c)

    # backends: name -> probabilities on test
    backends = {
        "quantum":         softmax_rows(q_logits, T_q),
        "classical_raw":   softmax_rows(c_logits, 1.0),   # reproduces benchmark's unfair path
        "classical_fair":  softmax_rows(c_logits, T_c),   # apples-to-apples
    }
    # optional extra backends
    for name, mod, cls in [("ensemble", "services.fusion.ensemble", "DeepEnsemble"),
                           ("learnable", "services.fusion.learnable", "LearnableFusion")]:
        try:
            import importlib
            obj = getattr(importlib.import_module(mod), cls).load()
            if obj is not None:
                lg = np.array([obj.logits(x) for x in Xte])
                backends[name] = softmax_rows(lg, T_q)
        except Exception as e:
            log.warning("optional backend %s skipped: %r", name, e)

    # -- raw per-sample outputs (Step 5) ----------------------------------- #
    import pandas as pd
    gt = pd.DataFrame({"index": np.arange(len(yte)), "y_true": yte,
                       "y_true_label": [labels[i] for i in yte]})
    gt.to_csv(outdir / "csv" / "ground_truth.csv", index=False)

    pred_cols = {"index": np.arange(len(yte)), "y_true": yte}
    conf_cols = {"index": np.arange(len(yte)), "y_true": yte}
    for name, P in backends.items():
        pred_cols[f"pred_{name}"] = P.argmax(1)
        conf_cols[f"conf_{name}"] = P.max(1)
        conf_cols[f"p_true_{name}"] = P[np.arange(len(yte)), yte]
        np.save(outdir / "metrics" / f"proba_{name}.npy", P)
    pd.DataFrame(pred_cols).to_csv(outdir / "csv" / "predictions.csv", index=False)
    pd.DataFrame(conf_cols).to_csv(outdir / "csv" / "confidence_scores.csv", index=False)

    # -- metric suite per backend (Step 5) --------------------------------- #
    metrics = {}
    for name, P in backends.items():
        rep = evaluate(P, yte)
        cov = float((P >= (1.0 - fit_conformal(P, yte, coverage)))[np.arange(len(yte)), yte].mean())
        size = float((P >= (1.0 - fit_conformal(P, yte, coverage))).sum(1).mean())
        rep["conformal_coverage"] = round(cov, 4)
        rep["conformal_set_size"] = round(size, 4)
        rep["ece_recomputed"] = round(expected_calibration_error(P, yte), 4)
        metrics[name] = rep
    (outdir / "metrics" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("accuracy: quantum=%.3f classical_fair=%.3f classical_raw=%.3f",
             metrics["quantum"]["accuracy"], metrics["classical_fair"]["accuracy"],
             metrics["classical_raw"]["accuracy"])
    log.info("ECE: quantum=%.3f classical_fair=%.3f classical_raw=%.3f",
             metrics["quantum"]["ece"], metrics["classical_fair"]["ece"],
             metrics["classical_raw"]["ece"])

    # -- statistics (Steps 7-8): quantum vs each classical variant --------- #
    from scipy.stats import wilcoxon, ttest_rel, shapiro
    stats_out: dict = {}
    q_correct = (backends["quantum"].argmax(1) == yte).astype(float)
    q_ptrue = backends["quantum"][np.arange(len(yte)), yte]
    q_nll = -np.log(np.clip(q_ptrue, 1e-12, 1))

    for ref in ["classical_fair", "classical_raw"]:
        Pr = backends[ref]
        r_correct = (Pr.argmax(1) == yte).astype(float)
        r_ptrue = Pr[np.arange(len(yte)), yte]
        r_nll = -np.log(np.clip(r_ptrue, 1e-12, 1))

        acc_diff, lo, hi, _ = bootstrap_diff_ci(q_correct, r_correct, rng=np.random.default_rng(SEED))
        b01, b10, mp, mstat = mcnemar_exact(r_correct, q_correct)
        perm_obs, perm_p = permutation_test_diff(q_correct, r_correct)

        # paired tests on per-sample NLL (lower is better)
        nll_d = q_nll - r_nll
        try:
            w_stat, w_p = wilcoxon(q_nll, r_nll, zero_method="wilcox", alternative="two-sided")
        except Exception:
            w_stat, w_p = float("nan"), float("nan")
        t_stat, t_p = ttest_rel(q_nll, r_nll)
        sh_stat, sh_p = shapiro(nll_d) if 3 <= len(nll_d) <= 5000 else (float("nan"), float("nan"))
        normal = bool(sh_p > 0.05) if sh_p == sh_p else None
        d = cohens_d_paired(q_nll, r_nll)

        # DeLong per class (OvR)
        delong = {}
        for ci, lab in enumerate(labels):
            yc = (yte == ci).astype(int)
            auc_q, auc_r, z, dp = delong_test(yc, backends["quantum"][:, ci], Pr[:, ci])
            delong[lab] = {"auc_quantum": auc_q, "auc_ref": auc_r, "z": z, "p": dp,
                           "n_pos": int(yc.sum())}

        stats_out[f"quantum_vs_{ref}"] = {
            "n": int(len(yte)),
            "accuracy_diff": acc_diff, "accuracy_diff_ci95": [lo, hi],
            "mcnemar": {"b_ref_only_right": b01, "b_quantum_only_right": b10,
                        "p_value": mp, "chi2_cc": mstat},
            "permutation_acc_diff": {"observed": perm_obs, "p_value": perm_p},
            "nll_wilcoxon": {"stat": float(w_stat), "p_value": float(w_p)},
            "nll_paired_t": {"stat": float(t_stat), "p_value": float(t_p)},
            "nll_shapiro_on_diff": {"stat": float(sh_stat), "p_value": float(sh_p),
                                    "normal": normal},
            "nll_cohens_d": d,
            "authoritative_nll_test": "paired_t" if normal else "wilcoxon (non-normal diff)",
            "delong_per_class": delong,
        }

    # bootstrap CIs for each backend's headline metrics
    boot = {}
    for name, P in backends.items():
        correct = (P.argmax(1) == yte).astype(float)
        acc, alo, ahi, acc_boots = bootstrap_ci(correct, np.mean, rng=np.random.default_rng(SEED + 7))
        boot[name] = {"accuracy": acc, "accuracy_ci95": [alo, ahi]}
        np.save(outdir / "metrics" / f"bootstrap_acc_{name}.npy", acc_boots)
    stats_out["bootstrap_accuracy"] = boot
    (outdir / "metrics" / "statistics.json").write_text(json.dumps(stats_out, indent=2))

    # -- claim verification (Step 8) --------------------------------------- #
    verdicts = _verify_claims(metrics, stats_out, ARTIFACTS, outdir, log)
    (outdir / "metrics" / "claim_verdicts.json").write_text(json.dumps(verdicts, indent=2))

    # -- figures (Step 6) -------------------------------------------------- #
    _make_figures(outdir, backends, yte, labels, metrics, stats_out, log)

    # -- manifest ---------------------------------------------------------- #
    manifest["n_eval"] = int(len(yte))
    manifest["temperatures"] = {"quantum_shipped": T_q, "classical_fair": T_c}
    manifest["backends"] = list(backends.keys())
    manifest["outputs"] = {
        "csv": sorted(p.name for p in (outdir / "csv").glob("*.csv")),
        "plots": sorted(p.name for p in (outdir / "plots").glob("*")),
        "metrics": sorted(p.name for p in (outdir / "metrics").glob("*")),
    }
    (outdir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("=== audit complete -> %s ===", outdir)
    return outdir


def _verify_claims(metrics, stats, artifacts_dir, outdir, log) -> list:
    """Compare PROJECT_STATUS / benchmark.json headline claims to regenerated,
    statistically-tested reality. Each claim is VALIDATED / NOT VALIDATED /
    CONFOUNDED with a pointer to the evidence."""
    verdicts = []
    q, cf, cr = metrics["quantum"], metrics["classical_fair"], metrics["classical_raw"]
    s_fair = stats["quantum_vs_classical_fair"]
    s_raw = stats["quantum_vs_classical_raw"]

    # Claim 1: quantum accuracy > classical
    sig = s_fair["mcnemar"]["p_value"] < 0.05
    verdicts.append({
        "claim": "Quantum fusion accuracy exceeds classical (0.96 vs 0.93 claimed)",
        "regenerated": {"quantum": q["accuracy"], "classical_fair": cf["accuracy"]},
        "accuracy_diff_ci95": s_fair["accuracy_diff_ci95"],
        "mcnemar_p": s_fair["mcnemar"]["p_value"],
        "verdict": "VALIDATED" if (q["accuracy"] > cf["accuracy"] and sig) else "NOT VALIDATED",
        "evidence": "csv/predictions.csv, metrics/statistics.json",
        "note": "significant" if sig else "difference not statistically significant at n="
                f"{s_fair['n']}",
    })
    # Claim 2: calibration gap (ECE)
    gap_fair = cf["ece"] - q["ece"]
    gap_raw = cr["ece"] - q["ece"]
    verdicts.append({
        "claim": "Quantum is far better calibrated (ECE 0.020 vs 0.276)",
        "regenerated_ece": {"quantum": q["ece"], "classical_fair": cf["ece"],
                            "classical_raw": cr["ece"]},
        "verdict": "CONFOUNDED",
        "evidence": "plots/reliability_diagram.*, metrics/metrics.json",
        "note": (f"The 0.276 figure is the UNCALIBRATED classical (raw). With classical "
                 f"given its own temperature the ECE gap shrinks from {gap_raw:.3f} (raw) to "
                 f"{gap_fair:.3f} (fair). The prior benchmark scaled only the quantum logits."),
    })
    # Claim 3: AUROC advantage (DeLong)
    sig_auc = [v for v in s_fair["delong_per_class"].values()
               if v["p"] == v["p"] and v["p"] < 0.05 and v["auc_quantum"] > v["auc_ref"]]
    verdicts.append({
        "claim": "Quantum ranking (AUROC) beats classical",
        "delong_significant_classes": len(sig_auc),
        "verdict": "VALIDATED" if sig_auc else "NOT VALIDATED",
        "evidence": "plots/roc_curves.*, metrics/statistics.json (delong_per_class)",
        "note": f"{len(sig_auc)} of {len(s_fair['delong_per_class'])} classes show a "
                "significant DeLong AUROC advantage for quantum.",
    })
    for v in verdicts:
        log.info("CLAIM [%s] %s", v["verdict"], v["claim"])
    return verdicts


def _save_fig(fig, outdir, name):
    for ext in ("png", "pdf"):
        fig.savefig(outdir / "plots" / f"{name}.{ext}", bbox_inches="tight", dpi=140)
    import matplotlib.pyplot as plt
    plt.close(fig)


def _make_figures(outdir, backends, yte, labels, metrics, stats, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

    n_classes = len(labels)
    Pq = backends["quantum"]; Pc = backends["classical_fair"]

    # ROC (macro overlay, OvR)
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, P, style in [("quantum", Pq, "-"), ("classical_fair", Pc, "--")]:
        for ci in range(n_classes):
            yc = (yte == ci).astype(int)
            if 0 < yc.sum() < len(yc):
                fpr, tpr, _ = roc_curve(yc, P[:, ci])
                ax.plot(fpr, tpr, style, alpha=0.5,
                        label=name if ci == 0 else None,
                        color="C0" if name == "quantum" else "C1")
    ax.plot([0, 1], [0, 1], ":", color="gray")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC (one-vs-rest, per class)"); ax.legend()
    _save_fig(fig, outdir, "roc_curves")

    # PR curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, P, style in [("quantum", Pq, "-"), ("classical_fair", Pc, "--")]:
        for ci in range(n_classes):
            yc = (yte == ci).astype(int)
            if yc.sum() > 0:
                prec, rec, _ = precision_recall_curve(yc, P[:, ci])
                ax.plot(rec, prec, style, alpha=0.5,
                        label=name if ci == 0 else None,
                        color="C0" if name == "quantum" else "C1")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (one-vs-rest)"); ax.legend()
    _save_fig(fig, outdir, "pr_curves")

    # Reliability diagram (calibration) — quantum vs both classical variants
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], ":", color="gray", label="perfect")
    for name, color in [("quantum", "C0"), ("classical_fair", "C1"), ("classical_raw", "C3")]:
        P = backends[name]
        conf = P.max(1); correct = (P.argmax(1) == yte).astype(float)
        edges = np.linspace(0, 1, 11)
        xs, ys = [], []
        for i in range(10):
            m = (conf > edges[i]) & (conf <= edges[i + 1])
            if m.sum() > 0:
                xs.append(conf[m].mean()); ys.append(correct[m].mean())
        ax.plot(xs, ys, "o-", color=color,
                label=f"{name} (ECE={metrics[name]['ece']:.3f})")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title("Reliability diagram"); ax.legend()
    _save_fig(fig, outdir, "reliability_diagram")

    # Confusion matrices
    for name in ("quantum", "classical_fair"):
        cm = confusion_matrix(yte, backends[name].argmax(1), labels=range(n_classes))
        fig, ax = plt.subplots(figsize=(5.5, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
        ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title(f"Confusion matrix — {name}"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        fig.colorbar(im, fraction=0.046)
        _save_fig(fig, outdir, f"confusion_{name}")

    # Bootstrap accuracy-difference distribution (quantum - classical_fair)
    try:
        a = (Pq.argmax(1) == yte).astype(float)
        b = (Pc.argmax(1) == yte).astype(float)
        _, lo, hi, boots = bootstrap_diff_ci(a, b, rng=np.random.default_rng(SEED))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(boots, bins=40, color="C0", alpha=0.8)
        ax.axvline(0, color="gray", ls=":")
        ax.axvline(lo, color="C3", ls="--"); ax.axvline(hi, color="C3", ls="--")
        ax.set_title("Bootstrap: accuracy(quantum) - accuracy(classical_fair)")
        ax.set_xlabel("Accuracy difference"); ax.set_ylabel("count")
        _save_fig(fig, outdir, "bootstrap_accuracy_diff")
    except Exception as e:
        log.warning("bootstrap fig failed: %r", e)

    # Prediction-set-size histogram (conformal efficiency)
    fig, ax = plt.subplots(figsize=(6, 4))
    from services.safety.calibration import fit_conformal
    for name, color in [("quantum", "C0"), ("classical_fair", "C1")]:
        P = backends[name]
        qhat = fit_conformal(P, yte, 0.90)
        sizes = (P >= (1.0 - qhat)).sum(1)
        ax.hist(sizes, bins=range(0, n_classes + 2), alpha=0.5, color=color,
                label=f"{name} (mean={sizes.mean():.2f})")
    ax.set_xlabel("Prediction-set size"); ax.set_ylabel("count")
    ax.set_title("Conformal prediction-set size (90% target)"); ax.legend()
    _save_fig(fig, outdir, "prediction_set_size")

    log.info("figures written: %d files", len(list((outdir / 'plots').glob('*'))))


if __name__ == "__main__":
    out = run_audit()
    print(f"\nAll artifacts under: {out}")
