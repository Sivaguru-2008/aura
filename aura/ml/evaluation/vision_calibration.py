"""Vision calibration & uncertainty suite (Step 4).

Calibrates and quantifies the uncertainty of the production DenseNet's multi-label
finding outputs on the MIMIC-CXR validation split:

    * Temperature scaling      — per-finding scalar T minimizing binary NLL, with
                                 ECE / reliability before vs after.
    * Monte-Carlo dropout      — stochastic forward passes (dropout kept active) for
                                 an epistemic-uncertainty estimate.
    * Test-time augmentation   — flip / intensity perturbation passes, an epistemic
                                 proxy that is meaningful even when the backbone has
                                 no dropout layers (deep-ensemble stand-in).
    * Conformal prediction     — per-finding split-conformal sets with empirical
                                 coverage and average set size (efficiency).
    * Reliability diagrams, confidence histograms, coverage plots.

Reuses ``services.safety`` primitives (``enable_dropout``, ``fit_temperature`` idea,
``binary_ece``) rather than reimplementing them. Artifacts land in
``artifacts/calibration/``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar

from common.config import ARTIFACTS
from schemas.clinical import FINDINGS
from ml.evaluation.clinical_eval import binary_ece, load_validation

FINDING_NAMES = [f.value for f in FINDINGS]
CAL_DIR = ARTIFACTS / "calibration"


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# --------------------------------------------------------------------------- #
# Inference returning logits
# --------------------------------------------------------------------------- #
def _infer_logits(image_paths, labels, batch_size: int = 32, model_path: Optional[str] = None):
    """Return (logits[N,7], y[N,7], VisionModel)."""
    import torch
    from torch.utils.data import DataLoader

    from ml.vision_cxr.dataset import ChestXrayDataset, get_transforms
    from ml.vision_cxr.inference import VisionModel

    vm = VisionModel(model_path or str(ARTIFACTS / "best_model.pt"))
    ds = ChestXrayDataset(image_paths, labels, transform=get_transforms(train=False))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    logits, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(vm.device, non_blocking=True)
            logits.append(vm.model(x).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(logits), np.concatenate(ys), vm


# --------------------------------------------------------------------------- #
# Temperature scaling (per finding)
# --------------------------------------------------------------------------- #
def fit_temperature_binary(logits_c: np.ndarray, y_c: np.ndarray) -> float:
    """One scalar T minimizing binary NLL of sigmoid(logit / T)."""
    logits_c = np.asarray(logits_c, float)
    y_c = np.asarray(y_c, float)

    def nll(logT: float) -> float:
        T = float(np.exp(logT))
        p = np.clip(_sigmoid(logits_c / T), 1e-9, 1 - 1e-9)
        return float(-np.mean(y_c * np.log(p) + (1 - y_c) * np.log(1 - p)))

    res = minimize_scalar(nll, bounds=(-2.0, 2.0), method="bounded")
    return float(np.exp(res.x))


def temperature_scaling(logits: np.ndarray, Y: np.ndarray) -> dict:
    """Fit per-finding temperatures on a calibration half, evaluate on the other."""
    n = len(Y)
    rng = np.random.default_rng(7)
    idx = rng.permutation(n)
    cut = n // 2
    cal, test = idx[:cut], idx[cut:]
    out = {"per_finding": {}, "temperatures": {}}
    ece_before, ece_after = [], []
    for c, name in enumerate(FINDING_NAMES):
        T = fit_temperature_binary(logits[cal, c], Y[cal, c])
        p_before = _sigmoid(logits[test, c])
        p_after = _sigmoid(logits[test, c] / T)
        eb = binary_ece(p_before, Y[test, c])
        ea = binary_ece(p_after, Y[test, c])
        out["temperatures"][name] = round(T, 4)
        out["per_finding"][name] = {"T": round(T, 4), "ece_before": round(eb, 4),
                                    "ece_after": round(ea, 4)}
        ece_before.append(eb); ece_after.append(ea)
    out["mean_ece_before"] = round(float(np.mean(ece_before)), 4)
    out["mean_ece_after"] = round(float(np.mean(ece_after)), 4)
    out["test_indices"] = test.tolist()
    return out


# --------------------------------------------------------------------------- #
# MC dropout + test-time augmentation (epistemic)
# --------------------------------------------------------------------------- #
def mc_dropout_uncertainty(vm, image_paths, k: int = 20, batch_size: int = 16,
                           limit: int = 64) -> dict:
    """Epistemic std per finding from K dropout-active passes (+ layer count)."""
    import torch
    from torch.utils.data import DataLoader

    from ml.vision_cxr.dataset import ChestXrayDataset, get_transforms
    from services.safety.uncertainty import enable_dropout

    n_dropout = sum(1 for m in vm.model.modules()
                    if m.__class__.__name__.startswith("Dropout"))
    subset = image_paths[:limit]
    ds = ChestXrayDataset(subset, np.zeros((len(subset), len(FINDING_NAMES)), np.float32),
                          transform=get_transforms(train=False))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    vm.model.eval()
    enable_dropout(vm.model)
    passes = []
    with torch.no_grad():
        for _ in range(k):
            batch_p = []
            for x, _ in loader:
                x = x.to(vm.device)
                batch_p.append(torch.sigmoid(vm.model(x)).cpu().numpy())
            passes.append(np.concatenate(batch_p))
    vm.model.eval()
    stack = np.stack(passes)                              # (k, N, C)
    std = stack.std(0).mean(0)                            # per finding
    return {
        "method": "mc_dropout",
        "n_dropout_layers": int(n_dropout),
        "passes": k,
        "n_images": len(subset),
        "epistemic_std": {FINDING_NAMES[c]: round(float(std[c]), 5) for c in range(len(FINDING_NAMES))},
        "note": ("architecture has no dropout layers; std ~0 - see test_time_augmentation"
                 if n_dropout == 0 else "dropout-based epistemic estimate"),
    }


def tta_uncertainty(vm, image_paths, k: int = 12, batch_size: int = 16, limit: int = 64) -> dict:
    """Test-time-augmentation epistemic proxy (flip + intensity jitter)."""
    import cv2
    import torch

    rng = np.random.default_rng(7)
    subset = image_paths[:limit]
    imgs = []
    for p in subset:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            g = np.zeros((224, 224), np.uint8)
        imgs.append(cv2.resize(g, (224, 224)).astype(np.float32) / 255.0)
    passes = []
    with torch.no_grad():
        for _ in range(k):
            batch = []
            for g in imgs:
                aug = g.copy()
                if rng.random() < 0.5:
                    aug = aug[:, ::-1].copy()
                aug = np.clip(aug * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05), 0, 1)
                t = torch.from_numpy((aug - 0.449) / 0.226).unsqueeze(0).unsqueeze(0)
                batch.append(t)
            xb = torch.cat(batch).to(vm.device)
            passes.append(torch.sigmoid(vm.model(xb)).cpu().numpy())
    stack = np.stack(passes)
    std = stack.std(0).mean(0)
    return {
        "method": "test_time_augmentation",
        "passes": k,
        "n_images": len(subset),
        "epistemic_std": {FINDING_NAMES[c]: round(float(std[c]), 5) for c in range(len(FINDING_NAMES))},
        "mean_epistemic_std": round(float(std.mean()), 5),
    }


# --------------------------------------------------------------------------- #
# Conformal prediction (per finding)
# --------------------------------------------------------------------------- #
def conformal_evaluation(probs: np.ndarray, Y: np.ndarray, coverage: float = 0.9) -> dict:
    """Per-finding split-conformal: coverage of the true binary label + set size."""
    n = len(Y)
    rng = np.random.default_rng(11)
    idx = rng.permutation(n)
    cut = n // 2
    cal, test = idx[:cut], idx[cut:]
    per = {}
    covs, sizes = [], []
    for c, name in enumerate(FINDING_NAMES):
        p = probs[:, c]
        # nonconformity of the true label: 1 - p_trueclass
        s_cal = np.where(Y[cal, c] == 1, 1 - p[cal], p[cal])
        m = len(s_cal)
        level = min(1.0, np.ceil((m + 1) * coverage) / m)
        qhat = float(np.quantile(s_cal, level, method="higher"))
        # test sets: include label L if (1 - p_L) <= qhat
        covered, set_sizes = 0, []
        for i in test:
            admit = []
            if (1 - p[i]) <= qhat:
                admit.append(1)
            if p[i] <= qhat:
                admit.append(0)
            if not admit:
                admit = [int(p[i] >= 0.5)]
            set_sizes.append(len(admit))
            if int(Y[i, c]) in admit:
                covered += 1
        cov = covered / len(test)
        avg = float(np.mean(set_sizes))
        per[name] = {"qhat": round(qhat, 4), "empirical_coverage": round(cov, 4),
                     "avg_set_size": round(avg, 4)}
        covs.append(cov); sizes.append(avg)
    return {
        "target_coverage": coverage,
        "per_finding": per,
        "mean_coverage": round(float(np.mean(covs)), 4),
        "mean_set_size": round(float(np.mean(sizes)), 4),
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _plots(logits, probs, Y, temp, out_dir: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    C = Y.shape[1]
    ncol, nrow = 4, int(np.ceil(C / 4))
    written = {}

    # Reliability before vs after temperature scaling
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    Ts = temp["temperatures"]
    for c in range(C):
        ax = axes[c]
        p_b = probs[:, c]
        p_a = _sigmoid(logits[:, c] / Ts[FINDING_NAMES[c]])
        for p, style, lab in ((p_b, "o-", "raw"), (p_a, "s--", "scaled")):
            edges = np.linspace(0, 1, 11)
            xs, ys = [], []
            for i in range(10):
                m = (p > edges[i]) & (p <= edges[i + 1])
                if m.sum():
                    xs.append(p[m].mean()); ys.append(Y[m, c].mean())
            ax.plot(xs, ys, style, ms=3, lw=1.1, label=lab)
        ax.plot([0, 1], [0, 1], "--", color="#888", lw=0.8)
        ax.set_title(FINDING_NAMES[c], fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        if c == 0:
            ax.legend(fontsize=6)
    for k in range(C, len(axes)):
        axes[k].axis("off")
    fig.suptitle("Reliability: raw vs temperature-scaled", fontsize=11)
    fig.tight_layout(); p = out_dir / "reliability_temperature.png"
    fig.savefig(p, dpi=140); plt.close(fig); written["reliability_temperature"] = str(p)

    # Confidence histograms
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for c in range(C):
        axes[c].hist(probs[:, c], bins=20, color="#a78bfa", alpha=0.85)
        axes[c].set_title(FINDING_NAMES[c], fontsize=8)
    for k in range(C, len(axes)):
        axes[k].axis("off")
    fig.suptitle("Confidence histograms", fontsize=11)
    fig.tight_layout(); p = out_dir / "confidence_histograms.png"
    fig.savefig(p, dpi=140); plt.close(fig); written["confidence_histograms"] = str(p)
    return written


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_calibration(limit: Optional[int] = None, make_plots: bool = True,
                    mc_passes: int = 20, out_dir: Optional[Path] = None,
                    model_path: Optional[str] = None) -> dict:
    out_dir = Path(out_dir) if out_dir else CAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths, labels = load_validation(limit=limit)
    if len(image_paths) == 0:
        raise RuntimeError("No validation images found on disk (check MIMIC paths).")
    logits, Y, vm = _infer_logits(image_paths, labels, model_path=model_path)
    probs = _sigmoid(logits)

    temp = temperature_scaling(logits, Y)
    conformal = conformal_evaluation(probs, Y)
    mcd = mc_dropout_uncertainty(vm, image_paths, k=mc_passes)
    tta = tta_uncertainty(vm, image_paths)

    plots = _plots(logits, probs, Y, temp, out_dir / "plots") if make_plots else {}

    report = {
        "n_images": int(len(Y)),
        "temperature_scaling": temp,
        "conformal_prediction": conformal,
        "mc_dropout": mcd,
        "test_time_augmentation": tta,
        "deep_ensemble": {"available": False,
                          "note": "single checkpoint; MC-dropout / TTA used as the "
                                  "epistemic estimator (deep ensemble optional)."},
        "plots": plots,
    }
    (out_dir / "calibration.json").write_text(json.dumps(report, indent=2, default=str),
                                              encoding="utf-8")
    _summary_md(report, out_dir / "CALIBRATION_SUMMARY.md")
    return report


def _summary_md(rep: dict, path: Path) -> None:
    t = rep["temperature_scaling"]
    cf = rep["conformal_prediction"]
    L = ["# Vision Calibration & Uncertainty", "",
         f"- **Images:** {rep['n_images']}",
         f"- **Mean ECE:** {t['mean_ece_before']} → **{t['mean_ece_after']}** "
         f"after per-finding temperature scaling",
         f"- **Conformal coverage (target {cf['target_coverage']}):** "
         f"{cf['mean_coverage']} · mean set size {cf['mean_set_size']}",
         f"- **MC-dropout layers:** {rep['mc_dropout']['n_dropout_layers']} · "
         f"TTA mean epistemic std {rep['test_time_augmentation']['mean_epistemic_std']}", "",
         "## Per-finding temperature / ECE", "",
         "| finding | T | ECE before | ECE after | conformal cov | set size |",
         "|---|---|---|---|---|---|"]
    for n in t["per_finding"]:
        pf = t["per_finding"][n]
        c = cf["per_finding"][n]
        L.append(f"| {n} | {pf['T']} | {pf['ece_before']} | {pf['ece_after']} | "
                 f"{c['empirical_coverage']} | {c['avg_set_size']} |")
    path.write_text("\n".join(L), encoding="utf-8")


def run(limit: Optional[int] = None) -> dict:
    print("[calibrate] running vision calibration on MIMIC validation ...")
    rep = run_calibration(limit=limit)
    t = rep["temperature_scaling"]
    print(f"[calibrate] mean ECE {t['mean_ece_before']} -> {t['mean_ece_after']} "
          f"(temperature scaling); conformal coverage {rep['conformal_prediction']['mean_coverage']}")
    print(f"[calibrate] artifacts -> {CAL_DIR}")
    return rep


if __name__ == "__main__":
    run()
