import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import concurrent.futures
import json
import time
from pathlib import Path
import numpy as np
import scipy.stats as stats
import torch

from aura.common.config import ARTIFACTS
from aura.common.mathx import softmax
from aura.ml.evaluation.v6_confirmatory import torch_circuit
from aura.ml.evaluation.matched_optimiser import N_DX
from aura.services.safety.calibration import fit_temperature, expected_calibration_error

OUT_DIR = Path(__file__).resolve().parent
PART = ARTIFACTS / "v6"
SEEDS = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
MARGIN = 0.010

def load_data():
    dev = np.load(ARTIFACTS / "chexpert_evidence.npz")
    fresh = np.load(PART / "chexpert_fresh_evidence.npz")
    
    X_dev = dev["X"]
    y_dev = dev["y_u0"]
    
    perm = np.random.default_rng(7).permutation(len(X_dev))
    n_cal = int(0.20 * len(X_dev))
    cal_idx, tr_idx = perm[:n_cal], perm[n_cal:]
    
    Xtr, ytr = X_dev[tr_idx], y_dev[tr_idx]
    Xcal, ycal = X_dev[cal_idx], y_dev[cal_idx]
    
    Xte, yte = fresh["X"], fresh["y_u0"]
    
    return {
        "Xtr": Xtr, "ytr": ytr,
        "Xcal": Xcal, "ycal": ycal,
        "Xte": Xte, "yte": yte
    }

def train_vqc_seed(n_layers, scale, init_sd, lr, seed, Xtr, ytr, Xcal, ycal, max_epochs=150, patience=20, batch_size=50):
    torch.manual_seed(seed)
    circ = torch_circuit(8, n_layers, "ring")
    
    theta = torch.nn.Parameter(init_sd * torch.randn(n_layers, 8, 2, dtype=torch.float64))
    W = torch.nn.Parameter(0.1 * torch.randn(N_DX, 8, dtype=torch.float64))
    b = torch.nn.Parameter(torch.zeros(N_DX, dtype=torch.float64))
    params = [theta, W, b]
    
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    Xt = torch.tensor(Xtr, dtype=torch.float64)
    yt = torch.tensor(ytr, dtype=torch.long)
    Xc = torch.tensor(Xcal, dtype=torch.float64)
    yc = torch.tensor(ycal, dtype=torch.long)
    
    def forward(X):
        return circ(scale * X[:, :8], theta) @ W.T + b
        
    n = len(ytr)
    best_nll = float("inf")
    best_state = None
    best_ep = -1
    since = 0
    history = []
    
    rng = torch.Generator().manual_seed(seed)
    for ep in range(max_epochs):
        perm = torch.randperm(n, generator=rng)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(forward(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            
        with torch.no_grad():
            cal_nll = float(loss_fn(forward(Xc), yc).item())
            tr_nll = float(loss_fn(forward(Xt), yt).item())
        history.append({"epoch": ep, "train_nll": round(tr_nll, 6), "cal_nll": round(cal_nll, 6)})
        
        if cal_nll < best_nll - 1e-6:
            best_nll, best_ep, since = cal_nll, ep, 0
            best_state = [p.detach().clone() for p in params]
        else:
            since += 1
            if since >= patience:
                break
                
    for p, s in zip(params, best_state):
        p.data.copy_(s)
        
    return {
        "best_epoch": best_ep,
        "epochs_run": len(history),
        "best_calibration_nll": best_nll,
        "params": [p.detach().numpy() for p in params]
    }

def score_vqc(params, n_layers, scale, Xcal, ycal, Xte, yte):
    circ = torch_circuit(8, n_layers, "ring")
    theta_val, W_val, b_val = [torch.tensor(p) for p in params]
    
    def forward(X):
        return circ(scale * X[:, :8], theta_val) @ W_val.T + b_val
        
    with torch.no_grad():
        Lcal = forward(torch.tensor(Xcal, dtype=torch.float64)).numpy()
        
        # Batched evaluation on test set to avoid memory spikes
        bs = 4096
        Lte_list = []
        for i in range(0, len(Xte), bs):
            chunk = torch.tensor(Xte[i:i + bs], dtype=torch.float64)
            Lte_list.append(forward(chunk).numpy())
        Lte = np.vstack(Lte_list)
        
    T = fit_temperature(Lcal, ycal)
    P = np.array([softmax(row / T) for row in Lte])
    
    nll = -np.log(np.clip(P[np.arange(len(yte)), yte], 1e-12, 1.0))
    correct = (P.argmax(axis=1) == yte)
    acc = float(correct.mean())
    ece = float(expected_calibration_error(P, yte))
    
    return {
        "temperature": float(T),
        "nll": float(nll.mean()),
        "accuracy": acc,
        "ece": ece,
        "probs": P,
        "sample_nll": nll
    }

def run_worker_seed(arm_name, cfg, seed, data):
    n_layers, scale, init_sd, lr = cfg
    tag = f"{arm_name}_s{seed}"
    ckpt_path = OUT_DIR / f"{tag}_params.npz"
    probs_path = OUT_DIR / f"{tag}_eval.npz"
    
    if probs_path.exists():
        d = np.load(probs_path)
        return {
            "seed": seed,
            "nll": float(d["nll"]),
            "accuracy": float(d["accuracy"]),
            "ece": float(d["ece"]),
            "temperature": float(d["temperature"]),
        }
        
    # Check if seed 7 existing checkpoint exists
    if seed == 7:
        if arm_name == "c2_vqc":
            existing = PART / "C2_quantum_20.npz"
            if existing.exists():
                z = np.load(existing)
                params = [z["p0"], z["p1"], z["p2"]]
            else:
                params = None
        elif arm_name == "c1_vqc":
            existing = PART / "C1_y_u0_quantum_params.npz"
            if existing.exists():
                z = np.load(existing)
                params = [z["p0"], z["p1"], z["p2"]]
            else:
                params = None
        else:
            params = None
    else:
        params = None
        
    if params is None:
        if ckpt_path.exists():
            z = np.load(ckpt_path)
            params = [z["p0"], z["p1"], z["p2"]]
        else:
            t0 = time.time()
            res = train_vqc_seed(n_layers, scale, init_sd, lr, seed,
                                 data["Xtr"], data["ytr"], data["Xcal"], data["ycal"])
            params = res["params"]
            np.savez(ckpt_path, p0=params[0], p1=params[1], p2=params[2])
            print(f"[{arm_name}:seed{seed}] Trained in {time.time() - t0:.1f}s, cal_nll={res['best_calibration_nll']:.4f}", flush=True)
            
    eval_res = score_vqc(params, n_layers, scale, data["Xcal"], data["ycal"], data["Xte"], data["yte"])
    np.savez_compressed(probs_path,
                        nll=eval_res["nll"],
                        accuracy=eval_res["accuracy"],
                        ece=eval_res["ece"],
                        temperature=eval_res["temperature"],
                        probs=eval_res["probs"],
                        sample_nll=eval_res["sample_nll"])
    return {
        "seed": seed,
        "nll": eval_res["nll"],
        "accuracy": eval_res["accuracy"],
        "ece": eval_res["ece"],
        "temperature": eval_res["temperature"],
    }

def main():
    print("Loading CheXpert evidence data...", flush=True)
    data = load_data()
    print(f"Dev train: {len(data['ytr'])}, Cal: {len(data['ycal'])}, Test: {len(data['yte'])}", flush=True)
    
    # Configurations
    c2_vqc_cfg = (4, 0.5, 0.01, 0.01)     # Winner: L=4, s=0.5, sd=0.01, lr=0.01
    c1_vqc_cfg = (3, 1.0, 0.1, 0.05)      # Default: L=3, s=1.0, sd=0.1, lr=0.05
    
    # Run C1 VQC (faster, ~3 mins per seed)
    print("\n--- Training & Evaluating C1 Default VQC across 10 seeds ---", flush=True)
    c1_vqc_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(run_worker_seed, "c1_vqc", c1_vqc_cfg, s, data): s for s in SEEDS}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            c1_vqc_results.append(res)
            print(f"  [C1 VQC Seed {res['seed']}] Test NLL={res['nll']:.5f}, Acc={res['accuracy']:.4f}, ECE={res['ece']:.4f}", flush=True)
    c1_vqc_results.sort(key=lambda x: x["seed"])
    
    # Run C2 VQC
    print("\n--- Training & Evaluating C2 Winner VQC across 10 seeds ---", flush=True)
    c2_vqc_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(run_worker_seed, "c2_vqc", c2_vqc_cfg, s, data): s for s in SEEDS}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            c2_vqc_results.append(res)
            print(f"  [C2 VQC Seed {res['seed']}] Test NLL={res['nll']:.5f}, Acc={res['accuracy']:.4f}, ECE={res['ece']:.4f}", flush=True)
    c2_vqc_results.sort(key=lambda x: x["seed"])
    
    # Load Classical baselines
    classical_summary = json.loads((OUT_DIR / "classical_seeds_summary.json").read_text())
    c2_poe_results = sorted([{"nll": r["test_nll"], **r} for r in classical_summary["c2_tuned_poe"]], key=lambda x: x["seed"])
    c2_gated_results = sorted([{"nll": r["test_nll"], **r} for r in classical_summary["c2_tuned_gated"]], key=lambda x: x["seed"])
    c1_poe_results = sorted([{"nll": r["test_nll"], **r} for r in classical_summary["c1_default_poe"]], key=lambda x: x["seed"])
    
    # Full statistical analysis
    def analyze_arm(arr):
        vals = [r["nll"] for r in arr]
        return {
            "mean": float(np.mean(vals)),
            "sd": float(np.std(vals, ddof=1)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "values": vals
        }
        
    def paired_inference(q_arr, c_arr, name):
        deltas = np.array([q["nll"] - c["nll"] for q, c in zip(q_arr, c_arr)])
        mean_d = float(np.mean(deltas))
        sd_d = float(np.std(deltas, ddof=1))
        n = len(deltas)
        se = sd_d / np.sqrt(n)
        df = n - 1
        
        t_crit = float(stats.t.ppf(0.975, df))
        ci95 = [float(mean_d - t_crit * se), float(mean_d + t_crit * se)]
        t_stat = mean_d / se
        p_two_sided = float(2 * (1 - stats.t.cdf(abs(t_stat), df)))
        
        # TOST vs +/- MARGIN (0.010)
        t_lower = (mean_d - (-MARGIN)) / se
        t_upper = (MARGIN - mean_d) / se
        p_lower = float(1 - stats.t.cdf(t_lower, df))
        p_upper = float(1 - stats.t.cdf(t_upper, df))
        p_tost = max(p_lower, p_upper)
        
        t_crit_90 = float(stats.t.ppf(0.95, df))
        ci90 = [float(mean_d - t_crit_90 * se), float(mean_d + t_crit_90 * se)]
        equivalent = bool(ci90[0] >= -MARGIN and ci90[1] <= MARGIN and p_tost < 0.05)
        
        return {
            "name": name,
            "mean_delta": mean_d,
            "sd_delta": sd_d,
            "se_delta": se,
            "df": df,
            "ci95_student_t": ci95,
            "t_stat_zero": t_stat,
            "p_two_sided": p_two_sided,
            "ci90_student_t": ci90,
            "tost": {
                "margin": MARGIN,
                "t_lower": t_lower,
                "p_lower": p_lower,
                "t_upper": t_upper,
                "p_upper": p_upper,
                "p_tost": p_tost,
                "equivalent": equivalent
            },
            "per_seed_deltas": deltas.tolist()
        }
        
    contrast_c2_poe = paired_inference(c2_vqc_results, c2_poe_results, "C2 VQC vs C2 PoE")
    contrast_c2_gated = paired_inference(c2_vqc_results, c2_gated_results, "C2 VQC vs C2 Gated Head")
    contrast_c1_poe = paired_inference(c1_vqc_results, c1_poe_results, "C1 default VQC vs C1 default PoE")
    
    # Hierarchical bootstrap for C2 VQC vs C2 PoE
    print("\nRunning hierarchical bootstrap (10,000 resamples)...", flush=True)
    rng = np.random.default_rng(7)
    
    def get_sample_nll(path):
        z = np.load(path)
        if "sample_nll" in z:
            return z["sample_nll"]
        if "pnll" in z:
            return z["pnll"]
        raise KeyError(f"No sample NLL found in {path}")

    vqc_sample_nlls = np.vstack([get_sample_nll(OUT_DIR / f"c2_vqc_s{s}_eval.npz") for s in SEEDS])
    poe_sample_nlls = np.vstack([get_sample_nll(OUT_DIR / f"c2_tuned_poe_seed{s}_eval.npz") for s in SEEDS])
    delta_sample_nlls = vqc_sample_nlls - poe_sample_nlls  # shape (10, 56534)
    
    n_patients = delta_sample_nlls.shape[1]
    boot_means = []
    n_boot = 10_000
    for _ in range(n_boot):
        # 1. Resample seeds with replacement
        s_idx = rng.choice(10, size=10, replace=True)
        # Average across resampled seeds
        seed_mean_deltas = delta_sample_nlls[s_idx].mean(axis=0)
        # 2. Resample patients with replacement
        p_idx = rng.choice(n_patients, size=n_patients, replace=True)
        boot_means.append(float(seed_mean_deltas[p_idx].mean()))
    boot_means = np.array(boot_means)
    hierarchical_ci95 = [float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))]
    hierarchical_ci90 = [float(np.percentile(boot_means, 5.0)), float(np.percentile(boot_means, 95.0))]
    print(f"Hierarchical Bootstrap 95% CI: {hierarchical_ci95}", flush=True)
    
    # Recompute C1b seed-level t-test exactly
    c1b_path = PART / "C1b_partial.json"
    c1b_exact = {}
    if c1b_path.exists():
        c1b_data = json.loads(c1b_path.read_text())
        c1b_deltas = np.array([r["delta_nll"] for r in c1b_data])
        mean_c1b = float(np.mean(c1b_deltas))
        sd_c1b = float(np.std(c1b_deltas, ddof=1))
        se_c1b = sd_c1b / np.sqrt(len(c1b_deltas))
        t_c1b = mean_c1b / se_c1b
        p_c1b = float(2 * (1 - stats.t.cdf(abs(t_c1b), df=9)))
        c1b_exact = {
            "mean_delta": mean_c1b,
            "sd_delta": sd_c1b,
            "se_delta": se_c1b,
            "t_statistic": round(t_c1b, 4),
            "df": 9,
            "exact_p_value": p_c1b,
            "reported_in_paper": "p = 0.0013 (or 0.0020)",
            "correction_note": f"t = {t_c1b:.2f} with df=9 yields exact two-sided p = {p_c1b:.5f}"
        }
        print(f"\nC1b exact t-test: t={t_c1b:.4f}, df=9, p={p_c1b:.5f}", flush=True)
        
    # C1 gap vs sigma_seed
    c1_gap = 0.0205
    c1_vqc_sd = float(np.std([r["nll"] for r in c1_vqc_results], ddof=1))
    c1_ratio = c1_gap / c1_vqc_sd if c1_vqc_sd > 0 else 0.0
    
    summary = {
        "c2_vqc": {
            "config": list(c2_vqc_cfg),
            "summary": analyze_arm(c2_vqc_results),
            "per_seed": c2_vqc_results
        },
        "c1_vqc": {
            "config": list(c1_vqc_cfg),
            "summary": analyze_arm(c1_vqc_results),
            "per_seed": c1_vqc_results
        },
        "contrasts": {
            "c2_vqc_vs_poe": contrast_c2_poe,
            "c2_vqc_vs_gated": contrast_c2_gated,
            "c1_vqc_vs_poe": contrast_c1_poe,
        },
        "hierarchical_bootstrap_c2_vqc_vs_poe": {
            "ci95": hierarchical_ci95,
            "ci90": hierarchical_ci90,
            "n_boot": n_boot,
        },
        "c1b_exact": c1b_exact,
        "c1_gap_vs_sigma_seed": {
            "c1_single_seed_gap": c1_gap,
            "c1_vqc_sigma_seed": c1_vqc_sd,
            "ratio_gap_over_sigma": round(c1_ratio, 3),
            "statement": f"+0.0205 gap corresponds to {c1_ratio:.2f} sigma_seed"
        }
    }
    
    out_file = OUT_DIR / "a1_quantum_seeds_summary.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote full A1 summary to {out_file}", flush=True)

if __name__ == "__main__":
    main()
