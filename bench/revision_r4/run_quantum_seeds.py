import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch

from aura.common.mathx import softmax
from aura.ml.evaluation import v6_confirmatory as v6
from aura.ml.evaluation import matched_optimiser as mo
from aura.services.safety.calibration import fit_temperature, expected_calibration_error

OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]

CONFIGS = {
    "c2_winner_vqc": {
        "n_qubits": 8,
        "n_layers": 4,
        "entangler": "ring",
        "scale": 0.5,
        "sd": 0.01,
        "lr": 0.01,
        "wd": 0.0,
    },
    "c1_default_vqc": {
        "n_qubits": 8,
        "n_layers": 3,
        "entangler": "ring",
        "scale": 1.0,
        "sd": 0.1,
        "lr": 0.05,
        "wd": 0.0,
    },
}

def train_and_eval_seed(arm_name: str, s: int):
    eval_path = OUT_DIR / f"{arm_name}_seed{s}_eval.npz"
    param_path = OUT_DIR / f"{arm_name}_seed{s}_params.npz"

    if eval_path.exists() and param_path.exists():
        print(f"[{arm_name}:seed{s}] Already completed, loading cached.", flush=True)
        z = np.load(eval_path)
        meta = json.loads(str(z["meta"])) if "meta" in z else {}
        return meta

    torch.set_num_threads(2)
    cfg = CONFIGS[arm_name]
    Xtr, ytr, Xcal, ycal = v6.dev_splits()
    Xte, yte = v6.test_set()

    build = v6._build_quantum_fast(
        cfg["n_qubits"], cfg["n_layers"], cfg["entangler"], cfg["scale"], cfg["sd"]
    )

    t0 = time.time()
    mo.SEED = s
    mo.WEIGHT_DECAY = cfg["wd"]
    mo.MAX_EPOCHS = 150
    mo.PATIENCE = 20

    print(f"[{arm_name}:seed{s}] Starting training...", flush=True)
    rec, params, _ = mo._train_under_protocol(build, Xtr, ytr, Xcal, ycal, cfg["lr"])
    v6._save_params(param_path, params)
    train_time = time.time() - t0

    fwd = v6._load_forward(build, param_path)
    Lcal = v6._logits(fwd, Xcal)
    Lte = v6._logits(fwd, Xte)

    T = fit_temperature(Lcal, ycal)
    P = np.array([softmax(r / T) for r in Lte])
    pnll = -np.log(np.clip(P[np.arange(len(yte)), yte], 1e-12, 1))
    acc = float((P.argmax(1) == yte).mean())
    ece = float(expected_calibration_error(P, yte))

    res = {
        "arm": arm_name,
        "seed": s,
        "config": cfg,
        "best_calibration_nll": rec["best_calibration_nll"],
        "best_epoch": rec["best_epoch"],
        "epochs_run": rec["epochs_run"],
        "train_seconds": round(train_time, 1),
        "temperature": round(float(T), 4),
        "test_nll": round(float(pnll.mean()), 5),
        "test_accuracy": round(acc, 5),
        "test_ece": round(ece, 5),
    }

    np.savez(
        eval_path,
        Lcal=Lcal,
        Lte=Lte,
        probs=P,
        pnll=pnll,
        T=T,
        meta=json.dumps(res),
    )
    print(
        f"[{arm_name}:seed{s}] Done: test_nll={res['test_nll']:.5f}, "
        f"acc={res['test_accuracy']:.4f}, cal_nll={res['best_calibration_nll']:.4f}, "
        f"epochs={res['epochs_run']} ({res['train_seconds']}s)",
        flush=True,
    )
    return res

def main():
    print(f"Starting quantum seed runs with up to 8 workers on {os.cpu_count()} cores...", flush=True)
    t_start = time.time()

    tasks = []
    # c2_winner_vqc first, then c1_default_vqc
    for arm_name in ["c2_winner_vqc", "c1_default_vqc"]:
        for s in SEEDS:
            tasks.append((arm_name, s))

    results = {"c2_winner_vqc": [], "c1_default_vqc": []}

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(train_and_eval_seed, arm, s): (arm, s) for arm, s in tasks}
        for future in concurrent.futures.as_completed(futures):
            arm, s = futures[future]
            try:
                res = future.result()
                results[arm].append(res)
            except Exception as e:
                print(f"FAILED {arm} seed {s}: {e}", flush=True)
                import traceback
                traceback.print_exc()

    # Sort by seed
    for k in results:
        results[k].sort(key=lambda r: r["seed"])

    (OUT_DIR / "quantum_seeds_summary.json").write_text(json.dumps(results, indent=2))
    print(f"\nAll quantum seed runs completed in {time.time() - t_start:.1f}s", flush=True)

if __name__ == "__main__":
    main()
