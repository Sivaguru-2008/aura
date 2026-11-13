import time
import json
from pathlib import Path
import numpy as np
import torch
from aura.ml.evaluation import v6_confirmatory as v6
from aura.ml.evaluation import matched_optimiser as mo
from aura.common.mathx import softmax
from aura.services.safety.calibration import fit_temperature, expected_calibration_error

OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

Xtr, ytr, Xcal, ycal = v6.dev_splits()
Xte, yte = v6.test_set()
d = Xtr.shape[1]

SEEDS = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]

print(f"Data loaded: train {len(ytr)}, cal {len(ycal)}, test {len(yte)}")

t0 = time.time()
results = {}

for arm_name, (build_fn, lr, wd) in [
    ("c2_tuned_poe", (v6._build_linear_c(d, 0.01), 0.05, 1e-4)),
    ("c2_tuned_gated", (v6._build_gated_c(d, 0.01), 0.1, 1e-4)),
    ("c1_default_poe", (mo._build_linear(d), 0.05, 0.0)),
]:
    arm_seeds = []
    print(f"\nRunning {arm_name} across 10 seeds...")
    for s in SEEDS:
        mo.SEED = s
        mo.WEIGHT_DECAY = wd
        rec, params, _ = mo._train_under_protocol(build_fn, Xtr, ytr, Xcal, ycal, lr)
        param_path = OUT_DIR / f"{arm_name}_seed{s}_params.npz"
        v6._save_params(param_path, params)
        fwd = v6._load_forward(build_fn, param_path)
        Lcal = v6._logits(fwd, Xcal)
        Lte = v6._logits(fwd, Xte)
        T = fit_temperature(Lcal, ycal)
        P = np.array([softmax(r / T) for r in Lte])
        pnll = -np.log(np.clip(P[np.arange(len(yte)), yte], 1e-12, 1))
        acc = float((P.argmax(1) == yte).mean())
        ece = float(expected_calibration_error(P, yte))
        np.savez(OUT_DIR / f"{arm_name}_seed{s}_eval.npz", Lcal=Lcal, Lte=Lte, probs=P, pnll=pnll, T=T)
        arm_seeds.append({
            "seed": s,
            "best_calibration_nll": rec["best_calibration_nll"],
            "best_epoch": rec["best_epoch"],
            "epochs_run": rec["epochs_run"],
            "temperature": round(float(T), 4),
            "test_nll": round(float(pnll.mean()), 5),
            "test_accuracy": round(acc, 5),
            "test_ece": round(ece, 5),
        })
        print(f"  seed {s}: test NLL = {arm_seeds[-1]['test_nll']:.5f}, cal NLL = {arm_seeds[-1]['best_calibration_nll']:.5f}, T = {arm_seeds[-1]['temperature']:.4f}")
    results[arm_name] = arm_seeds

(OUT_DIR / "classical_seeds_summary.json").write_text(json.dumps(results, indent=2))
print(f"\nAll classical arms finished in {time.time() - t0:.1f}s")
