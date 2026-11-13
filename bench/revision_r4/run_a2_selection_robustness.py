import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import time
from pathlib import Path
import numpy as np
import scipy.stats as stats
import torch

from aura.common.config import ARTIFACTS
from aura.common.mathx import softmax
from aura.ml.evaluation.v6_confirmatory import torch_circuit, Q_SPACE, C_SPACE, BUDGET
from aura.ml.evaluation.matched_optimiser import N_DX
import aura.ml.evaluation.matched_optimiser as mo
from aura.services.safety.calibration import fit_temperature, expected_calibration_error

OUT_DIR = Path(__file__).resolve().parent
PART = ARTIFACTS / "v6"
N_RESAMPLES = 20
MARGIN = 0.010

def load_data():
    dev = np.load(ARTIFACTS / "chexpert_evidence.npz")
    fresh = np.load(PART / "chexpert_fresh_evidence.npz")
    return {
        "X_dev": dev["X"], "y_dev": dev["y_u0"],
        "X_test": fresh["X"], "y_test": fresh["y_u0"]
    }

def score_logits(L, y, T):
    P = np.array([softmax(row / T) for row in L])
    nll = float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0)).mean())
    acc = float((P.argmax(axis=1) == y).mean())
    ece = float(expected_calibration_error(P, y))
    return nll, acc, ece

def precompute_arm_logits(arm_name, configs, data):
    X_dev = data["X_dev"]
    X_test = data["X_test"]
    n_dev = len(X_dev)
    n_test = len(X_test)
    
    print(f"Precomputing logits for {arm_name} (36 models)...", flush=True)
    logits_dev = []
    logits_test = []
    
    bs = 4096
    
    for i, cfg in enumerate(configs):
        ckpt_path = PART / f"C2_{arm_name}_{i:02d}.npz"
        z = np.load(ckpt_path)
        
        if arm_name == "quantum":
            L_val, s_val, sd_val, lr_val = cfg
            circ = torch_circuit(8, L_val, "ring")
            th = torch.tensor(z["p0"], dtype=torch.float64)
            W = torch.tensor(z["p1"], dtype=torch.float64)
            b = torch.tensor(z["p2"], dtype=torch.float64)
            
            with torch.no_grad():
                ld = (circ(s_val * torch.tensor(X_dev[:, :8], dtype=torch.float64), th) @ W.T + b).numpy()
                lt_list = []
                for start in range(0, n_test, bs):
                    chunk = torch.tensor(X_test[start:start+bs, :8], dtype=torch.float64)
                    lt_list.append((circ(s_val * chunk, th) @ W.T + b).numpy())
                lt = np.vstack(lt_list)
        elif arm_name == "classical":
            W = torch.tensor(z["p0"], dtype=torch.float64)
            b = torch.tensor(z["p1"], dtype=torch.float64)
            with torch.no_grad():
                ld = (torch.tensor(X_dev, dtype=torch.float64) @ W.T + b).numpy()
                lt = (torch.tensor(X_test, dtype=torch.float64) @ W.T + b).numpy()
        elif arm_name == "learnable":
            A = torch.tensor(z["p0"], dtype=torch.float64)
            c = torch.tensor(z["p1"], dtype=torch.float64)
            W = torch.tensor(z["p2"], dtype=torch.float64)
            b = torch.tensor(z["p3"], dtype=torch.float64)
            with torch.no_grad():
                Xd = torch.tensor(X_dev, dtype=torch.float64)
                ld = ((torch.sigmoid(Xd @ A.T + c) * Xd) @ W.T + b).numpy()
                
                lt_list = []
                for start in range(0, n_test, bs):
                    chunk = torch.tensor(X_test[start:start+bs], dtype=torch.float64)
                    lt_list.append(((torch.sigmoid(chunk @ A.T + c) * chunk) @ W.T + b).numpy())
                lt = np.vstack(lt_list)
        else:
            raise ValueError(arm_name)
            
        logits_dev.append(ld)
        logits_test.append(lt)
        
    return logits_dev, logits_test

def run_resampled_selection(arm_logits, data, configs_per_arm):
    X_dev, y_dev = data["X_dev"], data["y_dev"]
    X_test, y_test = data["X_test"], data["y_test"]
    n_dev = len(y_dev)
    n_cal = int(0.20 * n_dev)
    
    arms = ["quantum", "classical", "learnable"]
    resample_records = []
    
    print(f"\nRunning selection across {N_RESAMPLES} resampled dev splits...", flush=True)
    
    for split_id in range(N_RESAMPLES):
        seed = 100 + split_id
        perm = np.random.default_rng(seed).permutation(n_dev)
        cal_idx = perm[:n_cal]
        y_cal = y_dev[cal_idx]
        
        split_record = {"split_id": split_id, "seed": seed, "arms": {}}
        
        for arm in arms:
            dev_l_list, test_l_list = arm_logits[arm]
            cfgs = configs_per_arm[arm]
            
            best_cal_nll = float("inf")
            best_idx = -1
            best_T = 1.0
            
            for i in range(36):
                L_cal = dev_l_list[i][cal_idx]
                T = fit_temperature(L_cal, y_cal)
                nll_cal, _, _ = score_logits(L_cal, y_cal, T)
                if nll_cal < best_cal_nll:
                    best_cal_nll = nll_cal
                    best_idx = i
                    best_T = T
                    
            winner_cfg = cfgs[best_idx]
            L_test = test_l_list[best_idx]
            test_nll, test_acc, test_ece = score_logits(L_test, y_test, best_T)
            
            split_record["arms"][arm] = {
                "winner_idx": best_idx,
                "winner_config": list(winner_cfg),
                "winner_file": f"C2_{arm}_{best_idx:02d}.npz",
                "calibration_nll": round(best_cal_nll, 5),
                "temperature": round(best_T, 4),
                "test_nll": round(test_nll, 5),
                "test_accuracy": round(test_acc, 5),
                "test_ece": round(test_ece, 5)
            }
            
        vqc_nll = split_record["arms"]["quantum"]["test_nll"]
        poe_nll = split_record["arms"]["classical"]["test_nll"]
        gat_nll = split_record["arms"]["learnable"]["test_nll"]
        
        split_record["delta_vqc_vs_poe"] = round(vqc_nll - poe_nll, 5)
        split_record["delta_vqc_vs_gated"] = round(vqc_nll - gat_nll, 5)
        resample_records.append(split_record)
        
        print(f"  [Split {split_id+1:02d}] Winner VQC idx={split_record['arms']['quantum']['winner_idx']:02d} (test NLL={vqc_nll:.5f}) | "
              f"Winner PoE idx={split_record['arms']['classical']['winner_idx']:02d} (test NLL={poe_nll:.5f}) | "
              f"Delta={split_record['delta_vqc_vs_poe']:+.5f}", flush=True)
              
    return resample_records

def evaluate_full_classical_grid(data):
    print("\n--- Sensitivity Analysis: Full Classical 60-Configuration Grid ---", flush=True)
    X_dev, y_dev = data["X_dev"], data["y_dev"]
    X_test, y_test = data["X_test"], data["y_test"]
    
    perm = np.random.default_rng(7).permutation(len(X_dev))
    n_cal = int(0.20 * len(X_dev))
    cal_idx, tr_idx = perm[:n_cal], perm[n_cal:]
    Xtr, ytr = X_dev[tr_idx], y_dev[tr_idx]
    Xcal, ycal = X_dev[cal_idx], y_dev[cal_idx]
    d = Xtr.shape[1]
    
    all_60_configs = C_SPACE
    drawn_36_indices = sorted(np.random.default_rng(7).choice(len(C_SPACE), BUDGET, replace=False).tolist())
    drawn_36_configs = [all_60_configs[i] for i in drawn_36_indices]
    
    # Train and evaluate all 60 configs
    full_results = []
    for idx, cfg in enumerate(all_60_configs):
        lr, wd, sd = cfg
        mo.SEED = 7
        mo.WEIGHT_DECAY = wd
        
        def build():
            torch.manual_seed(7)
            W = torch.nn.Parameter(sd * torch.randn(N_DX, d, dtype=torch.float64))
            b = torch.nn.Parameter(torch.zeros(N_DX, dtype=torch.float64))
            return [W, b], lambda X: X @ W.T + b
            
        rec, params, _ = mo._train_under_protocol(build, Xtr, ytr, Xcal, ycal, lr)
        W, b = params
        
        with torch.no_grad():
            Lcal = (torch.tensor(Xcal, dtype=torch.float64) @ W.T + b).numpy()
            Lte = (torch.tensor(X_test, dtype=torch.float64) @ W.T + b).numpy()
            
        T = fit_temperature(Lcal, ycal)
        cal_nll, _, _ = score_logits(Lcal, ycal, T)
        te_nll, te_acc, te_ece = score_logits(Lte, y_test, T)
        
        full_results.append({
            "grid_index": idx,
            "in_36_budget": bool(idx in drawn_36_indices),
            "config": list(cfg),
            "cal_nll": round(cal_nll, 5),
            "test_nll": round(te_nll, 5),
            "test_acc": round(te_acc, 5),
            "test_ece": round(te_ece, 5),
            "temperature": round(T, 4)
        })
        
    mo.WEIGHT_DECAY = 0.0
    
    # Winner among 36 budget
    budget_results = [r for r in full_results if r["in_36_budget"]]
    budget_winner = min(budget_results, key=lambda x: x["cal_nll"])
    
    # Winner among full 60
    full_winner = min(full_results, key=lambda x: x["cal_nll"])
    
    print(f"36-Budget Winner: {budget_winner['config']} | Cal NLL={budget_winner['cal_nll']:.5f} | Test NLL={budget_winner['test_nll']:.5f}", flush=True)
    print(f"Full-60 Winner:   {full_winner['config']} | Cal NLL={full_winner['cal_nll']:.5f} | Test NLL={full_winner['test_nll']:.5f}", flush=True)
    
    return {
        "total_configs": len(full_results),
        "budget_configs": len(budget_results),
        "budget_winner": budget_winner,
        "full_winner": full_winner,
        "winner_changed": bool(budget_winner["config"] != full_winner["config"]),
        "cal_nll_difference": round(budget_winner["cal_nll"] - full_winner["cal_nll"], 5),
        "test_nll_difference": round(budget_winner["test_nll"] - full_winner["test_nll"], 5),
        "all_60_configs": full_results
    }

def main():
    print("Loading data for A2 Selection Robustness...", flush=True)
    data = load_data()
    
    # Load configs from saved grid logs
    q_log = json.loads((PART / "C2_quantum_grid.json").read_text())
    c_log = json.loads((PART / "C2_classical_grid.json").read_text())
    g_log = json.loads((PART / "C2_learnable_grid.json").read_text())
    
    configs_per_arm = {
        "quantum": [r["config"] for r in q_log],
        "classical": [r["config"] for r in c_log],
        "learnable": [r["config"] for r in g_log]
    }
    
    arm_logits = {}
    for arm in ["quantum", "classical", "learnable"]:
        arm_logits[arm] = precompute_arm_logits(arm, configs_per_arm[arm], data)
        
    resamples = run_resampled_selection(arm_logits, data, configs_per_arm)
    
    # Summarize resamples
    def summarize_metric(vals):
        return {
            "mean": round(float(np.mean(vals)), 5),
            "sd": round(float(np.std(vals, ddof=1)), 5),
            "min": round(float(np.min(vals)), 5),
            "max": round(float(np.max(vals)), 5),
            "median": round(float(np.median(vals)), 5),
            "iqr": round(float(np.percentile(vals, 75) - np.percentile(vals, 25)), 5)
        }
        
    vqc_test_nlls = [r["arms"]["quantum"]["test_nll"] for r in resamples]
    poe_test_nlls = [r["arms"]["classical"]["test_nll"] for r in resamples]
    gat_test_nlls = [r["arms"]["learnable"]["test_nll"] for r in resamples]
    deltas_poe = [r["delta_vqc_vs_poe"] for r in resamples]
    deltas_gat = [r["delta_vqc_vs_gated"] for r in resamples]
    
    # Winner frequency
    vqc_winner_counts = {}
    for r in resamples:
        cfg_str = str(r["arms"]["quantum"]["winner_config"])
        vqc_winner_counts[cfg_str] = vqc_winner_counts.get(cfg_str, 0) + 1
        
    poe_winner_counts = {}
    for r in resamples:
        cfg_str = str(r["arms"]["classical"]["winner_config"])
        poe_winner_counts[cfg_str] = poe_winner_counts.get(cfg_str, 0) + 1
        
    # Sensitivity analysis
    sensitivity = evaluate_full_classical_grid(data)
    
    summary = {
        "n_resamples": N_RESAMPLES,
        "dev_split_partition": "80% train (6,400), 20% cal (1,600)",
        "test_evaluation": "56,534 held-out patients, scored without selection on it",
        "resample_summary": {
            "vqc_winner_test_nll": summarize_metric(vqc_test_nlls),
            "poe_winner_test_nll": summarize_metric(poe_test_nlls),
            "gated_winner_test_nll": summarize_metric(gat_test_nlls),
            "delta_vqc_vs_poe": summarize_metric(deltas_poe),
            "delta_vqc_vs_gated": summarize_metric(deltas_gat),
        },
        "winner_selection_stability": {
            "quantum_winners_frequency": vqc_winner_counts,
            "classical_winners_frequency": poe_winner_counts,
            "pre_registered_quantum_winner": [4, 0.5, 0.01, 0.01],
            "pre_registered_classical_winner": [0.05, 0.0001, 0.01],
        },
        "grid_draw_documentation": {
            "quantum": {
                "full_grid_size": len(Q_SPACE),
                "budget": BUDGET,
                "sampling_fraction": round(BUDGET / len(Q_SPACE), 4),
                "draw_mechanism": "Pseudorandom seeded draw: np.random.default_rng(7).choice(135, 36, replace=False)",
            },
            "classical": {
                "full_grid_size": len(C_SPACE),
                "budget": BUDGET,
                "sampling_fraction": round(BUDGET / len(C_SPACE), 4),
                "draw_mechanism": "Pseudorandom seeded draw: np.random.default_rng(7).choice(60, 36, replace=False)",
            }
        },
        "full_classical_grid_sensitivity": sensitivity,
        "resamples": resamples
    }
    
    out_file = OUT_DIR / "a2_selection_robustness.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote full A2 selection robustness results to {out_file}", flush=True)

if __name__ == "__main__":
    main()
