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
from scipy.optimize import minimize

from aura.common.config import ARTIFACTS
from aura.common.mathx import softmax
from aura.ml.evaluation.v6_confirmatory import torch_circuit
from aura.services.safety.calibration import fit_temperature, expected_calibration_error

OUT_DIR = Path(__file__).resolve().parent
PART = ARTIFACTS / "v6"
N_DX = 6
MARGIN = 0.010

ACTIVE_CLASSES = [
    (0, "Normal"),
    (5, "Pneumothorax"),
    (4, "Malignancy"),
    (1, "Pneumonia"),
    (2, "Heart Failure")
]

def load_data():
    dev = np.load(ARTIFACTS / "chexpert_evidence.npz")
    fresh = np.load(PART / "chexpert_fresh_evidence.npz")
    
    X_dev = dev["X"]
    y_dev = dev["y_u0"]
    
    perm = np.random.default_rng(7).permutation(len(X_dev))
    n_cal = int(0.20 * len(X_dev))
    cal_idx, tr_idx = perm[:n_cal], perm[n_cal:]
    
    Xcal, ycal = X_dev[cal_idx], y_dev[cal_idx]
    Xte, yte = fresh["X"], fresh["y_u0"]
    
    return Xcal, ycal, Xte, yte

def compute_logits(arm_name, Xcal, Xte):
    if arm_name == "vqc":
        # Winner: L=4, s=0.5
        circ = torch_circuit(8, 4, "ring")
        z = np.load(PART / "C2_quantum_20.npz")
        th, W, b = torch.tensor(z["p0"]), torch.tensor(z["p1"]), torch.tensor(z["p2"])
        def fwd(X):
            return (circ(0.5 * X[:, :8], th) @ W.T + b).numpy()
    elif arm_name == "poe":
        # Winner: sd=0.01, wd=1e-4, lr=0.05
        z = np.load(PART / "C2_classical_17.npz")
        W, b = torch.tensor(z["p0"]), torch.tensor(z["p1"])
        def fwd(X):
            return (X @ W.T + b).numpy()
    elif arm_name == "gated":
        # Winner: sd=0.01, wd=1e-4, lr=0.1
        z = np.load(PART / "C2_learnable_24.npz")
        A, c, W, b = torch.tensor(z["p0"]), torch.tensor(z["p1"]), torch.tensor(z["p2"]), torch.tensor(z["p3"])
        def fwd(X):
            return ((torch.sigmoid(X @ A.T + c) * X) @ W.T + b).numpy()
    else:
        raise ValueError(arm_name)
        
    with torch.no_grad():
        Lcal = fwd(torch.tensor(Xcal, dtype=torch.float64))
        
        # Batched test evaluation
        bs = 4096
        Lte_list = []
        for i in range(0, len(Xte), bs):
            Lte_list.append(fwd(torch.tensor(Xte[i:i + bs], dtype=torch.float64)))
        Lte = np.vstack(Lte_list)
        
    return Lcal, Lte

def fit_vector_scaling(Lcal, ycal):
    # Vector scaling: z_c = w_c * l_c + b_c (10 parameters)
    def loss_fn(params):
        W_diag = params[:N_DX]
        b = params[N_DX:]
        scaled = Lcal * W_diag + b
        # Softmax cross entropy
        max_s = np.max(scaled, axis=1, keepdims=True)
        exp_s = np.exp(scaled - max_s)
        p = exp_s / np.sum(exp_s, axis=1, keepdims=True)
        nll = -np.log(np.clip(p[np.arange(len(ycal)), ycal], 1e-12, 1.0))
        # L2 regularizer on W towards 1.0 and b towards 0.0
        reg = 1e-3 * np.sum((W_diag - 1.0)**2) + 1e-3 * np.sum(b**2)
        return float(np.mean(nll) + reg)
        
    init_params = np.concatenate([np.ones(N_DX), np.zeros(N_DX)])
    res = minimize(loss_fn, init_params, method="L-BFGS-B")
    return res.x[:N_DX], res.x[N_DX:]

def apply_vector_scaling(L, W_diag, b):
    scaled = L * W_diag + b
    max_s = np.max(scaled, axis=1, keepdims=True)
    exp_s = np.exp(scaled - max_s)
    return exp_s / np.sum(exp_s, axis=1, keepdims=True)

def main():
    print("Loading data for A3...", flush=True)
    Xcal, ycal, Xte, yte = load_data()
    n_test = len(yte)
    print(f"Loaded cal={len(ycal)}, test={n_test}", flush=True)
    
    print("Computing uncalibrated logits...", flush=True)
    Lcal_vqc, Lte_vqc = compute_logits("vqc", Xcal, Xte)
    Lcal_poe, Lte_poe = compute_logits("poe", Xcal, Xte)
    Lcal_gat, Lte_gat = compute_logits("gated", Xcal, Xte)
    
    # 1. Standard temperature scaling
    T_vqc = fit_temperature(Lcal_vqc, ycal)
    T_poe = fit_temperature(Lcal_poe, ycal)
    T_gat = fit_temperature(Lcal_gat, ycal)
    print(f"Fitted temperatures: VQC={T_vqc:.4f}, PoE={T_poe:.4f}, Gated={T_gat:.4f}", flush=True)
    
    P_vqc = np.array([softmax(r / T_vqc) for r in Lte_vqc])
    P_poe = np.array([softmax(r / T_poe) for r in Lte_poe])
    P_gat = np.array([softmax(r / T_gat) for r in Lte_gat])
    
    # Sample NLLs
    nll_vqc = -np.log(np.clip(P_vqc[np.arange(n_test), yte], 1e-12, 1.0))
    nll_poe = -np.log(np.clip(P_poe[np.arange(n_test), yte], 1e-12, 1.0))
    nll_gat = -np.log(np.clip(P_gat[np.arange(n_test), yte], 1e-12, 1.0))
    
    delta_poe = nll_vqc - nll_poe
    delta_gat = nll_vqc - nll_gat
    
    # Per-class analysis
    print("\n--- Per-Class Analysis ---", flush=True)
    # Bootstrap setup (10,000 resamples within each class)
    n_boot = 10_000
    rng = np.random.default_rng(7)
    
    per_class_table = []
    boot_class_deltas_poe = []
    boot_class_deltas_gat = []
    
    for c, dname in ACTIVE_CLASSES:
        idx = np.where(yte == c)[0]
        nc = len(idx)
        prev = nc / n_test
        
        vqc_c_nll = nll_vqc[idx]
        poe_c_nll = nll_poe[idx]
        gat_c_nll = nll_gat[idx]
        
        d_poe_c = delta_poe[idx]
        d_gat_c = delta_gat[idx]
        
        # Reliability
        mean_prob_vqc = float(P_vqc[:, c].mean())
        mean_prob_poe = float(P_poe[:, c].mean())
        mean_prob_gat = float(P_gat[:, c].mean())
        
        # Bootstrap within class
        boot_d_poe = []
        boot_d_gat = []
        for _ in range(n_boot):
            b_idx = rng.choice(nc, size=nc, replace=True)
            boot_d_poe.append(float(d_poe_c[b_idx].mean()))
            boot_d_gat.append(float(d_gat_c[b_idx].mean()))
            
        boot_d_poe = np.array(boot_d_poe)
        boot_d_gat = np.array(boot_d_gat)
        boot_class_deltas_poe.append(boot_d_poe)
        boot_class_deltas_gat.append(boot_d_gat)
        
        ci95_poe = [float(np.percentile(boot_d_poe, 2.5)), float(np.percentile(boot_d_poe, 97.5))]
        ci95_gat = [float(np.percentile(boot_d_gat, 2.5)), float(np.percentile(boot_d_gat, 97.5))]
        
        # Classwise ECE
        # Binary target: y == c
        bin_y = (yte == c).astype(int)
        ece_vqc_c = float(expected_calibration_error(np.column_stack([1 - P_vqc[:, c], P_vqc[:, c]]), bin_y))
        ece_poe_c = float(expected_calibration_error(np.column_stack([1 - P_poe[:, c], P_poe[:, c]]), bin_y))
        ece_gat_c = float(expected_calibration_error(np.column_stack([1 - P_gat[:, c], P_gat[:, c]]), bin_y))
        
        row = {
            "class_id": c,
            "diagnosis": dname,
            "n_patients": nc,
            "prevalence": round(prev, 4),
            "observed_rate": round(prev, 4),
            "mean_pred_prob": {
                "vqc": round(mean_prob_vqc, 4),
                "poe": round(mean_prob_poe, 4),
                "gated": round(mean_prob_gat, 4)
            },
            "mean_nll": {
                "vqc": round(float(vqc_c_nll.mean()), 5),
                "poe": round(float(poe_c_nll.mean()), 5),
                "gated": round(float(gat_c_nll.mean()), 5)
            },
            "classwise_ece": {
                "vqc": round(ece_vqc_c, 4),
                "poe": round(ece_poe_c, 4),
                "gated": round(ece_gat_c, 4)
            },
            "vqc_vs_poe": {
                "delta_nll": round(float(d_poe_c.mean()), 5),
                "ci95": [round(x, 5) for x in ci95_poe]
            },
            "vqc_vs_gated": {
                "delta_nll": round(float(d_gat_c.mean()), 5),
                "ci95": [round(x, 5) for x in ci95_gat]
            }
        }
        per_class_table.append(row)
        print(f"  [{dname:<14}] Prev={prev*100:5.2f}% | VQC-PoE dNLL={row['vqc_vs_poe']['delta_nll']:+.5f} 95% CI {ci95_poe}", flush=True)
        
    boot_class_deltas_poe = np.array(boot_class_deltas_poe) # shape (5, 10000)
    boot_class_deltas_gat = np.array(boot_class_deltas_gat)
    
    # Aggregate Metrics
    # 1. Prevalence-weighted (overall)
    prevs = np.array([r["prevalence"] for r in per_class_table])[:, None]
    prevs = prevs / np.sum(prevs)
    boot_prev_weighted_poe = np.sum(prevs * boot_class_deltas_poe, axis=0)
    boot_prev_weighted_gat = np.sum(prevs * boot_class_deltas_gat, axis=0)
    
    # 2. Class-balanced Macro-mean
    boot_macro_poe = np.mean(boot_class_deltas_poe, axis=0)
    boot_macro_gat = np.mean(boot_class_deltas_gat, axis=0)
    
    # 3. Worst-class Delta NLL
    obs_class_deltas_poe = np.array([r["vqc_vs_poe"]["delta_nll"] for r in per_class_table])
    obs_class_deltas_gat = np.array([r["vqc_vs_gated"]["delta_nll"] for r in per_class_table])
    
    worst_class_idx_poe = int(np.argmax(np.abs(obs_class_deltas_poe)))
    worst_class_idx_gat = int(np.argmax(np.abs(obs_class_deltas_gat)))
    
    # Worst class distribution
    boot_worst_poe = boot_class_deltas_poe[worst_class_idx_poe]
    boot_worst_gat = boot_class_deltas_gat[worst_class_idx_gat]
    
    # Macro TOST vs +/- 0.010
    macro_mean_poe = float(np.mean(boot_macro_poe))
    macro_ci90_poe = [float(np.percentile(boot_macro_poe, 5.0)), float(np.percentile(boot_macro_poe, 95.0))]
    macro_ci95_poe = [float(np.percentile(boot_macro_poe, 2.5)), float(np.percentile(boot_macro_poe, 97.5))]
    macro_equiv_poe = bool(macro_ci90_poe[0] >= -MARGIN and macro_ci90_poe[1] <= MARGIN)
    
    macro_mean_gat = float(np.mean(boot_macro_gat))
    macro_ci90_gat = [float(np.percentile(boot_macro_gat, 5.0)), float(np.percentile(boot_macro_gat, 95.0))]
    macro_ci95_gat = [float(np.percentile(boot_macro_gat, 2.5)), float(np.percentile(boot_macro_gat, 97.5))]
    macro_equiv_gat = bool(macro_ci90_gat[0] >= -MARGIN and macro_ci90_gat[1] <= MARGIN)
    
    print("\n--- Aggregate Metrics ---", flush=True)
    print(f"Prevalence-weighted dNLL (VQC vs PoE): {np.mean(boot_prev_weighted_poe):+.5f} 95% CI [{np.percentile(boot_prev_weighted_poe, 2.5):.5f}, {np.percentile(boot_prev_weighted_poe, 97.5):.5f}]", flush=True)
    print(f"Class-balanced Macro dNLL (VQC vs PoE): {macro_mean_poe:+.5f} 90% CI {macro_ci90_poe} | Equivalent? {macro_equiv_poe}", flush=True)
    print(f"Worst class ({per_class_table[worst_class_idx_poe]['diagnosis']}): {obs_class_deltas_poe[worst_class_idx_poe]:+.5f} 95% CI [{np.percentile(boot_worst_poe, 2.5):.5f}, {np.percentile(boot_worst_poe, 97.5):.5f}]", flush=True)
    
    # 2. Vector Scaling Ablation
    print("\n--- Fitting Vector Scaling Ablation ---", flush=True)
    w_vec_vqc, b_vec_vqc = fit_vector_scaling(Lcal_vqc, ycal)
    w_vec_poe, b_vec_poe = fit_vector_scaling(Lcal_poe, ycal)
    w_vec_gat, b_vec_gat = fit_vector_scaling(Lcal_gat, ycal)
    
    P_vec_vqc = apply_vector_scaling(Lte_vqc, w_vec_vqc, b_vec_vqc)
    P_vec_poe = apply_vector_scaling(Lte_poe, w_vec_poe, b_vec_poe)
    P_vec_gat = apply_vector_scaling(Lte_gat, w_vec_gat, b_vec_gat)
    
    nll_vec_vqc = float(-np.log(np.clip(P_vec_vqc[np.arange(n_test), yte], 1e-12, 1.0)).mean())
    nll_vec_poe = float(-np.log(np.clip(P_vec_poe[np.arange(n_test), yte], 1e-12, 1.0)).mean())
    nll_vec_gat = float(-np.log(np.clip(P_vec_gat[np.arange(n_test), yte], 1e-12, 1.0)).mean())
    
    ece_vec_vqc = float(expected_calibration_error(P_vec_vqc, yte))
    ece_vec_poe = float(expected_calibration_error(P_vec_poe, yte))
    ece_vec_gat = float(expected_calibration_error(P_vec_gat, yte))
    
    vector_scaling_summary = {
        "parameters": {
            "vqc": {"weights": [round(float(x), 4) for x in w_vec_vqc], "biases": [round(float(x), 4) for x in b_vec_vqc]},
            "poe": {"weights": [round(float(x), 4) for x in w_vec_poe], "biases": [round(float(x), 4) for x in b_vec_poe]},
            "gated": {"weights": [round(float(x), 4) for x in w_vec_gat], "biases": [round(float(x), 4) for x in b_vec_gat]},
        },
        "test_nll": {
            "vqc": round(nll_vec_vqc, 5),
            "poe": round(nll_vec_poe, 5),
            "gated": round(nll_vec_gat, 5),
            "delta_vqc_vs_poe": round(nll_vec_vqc - nll_vec_poe, 5),
            "delta_vqc_vs_gated": round(nll_vec_vqc - nll_vec_gat, 5)
        },
        "test_ece": {
            "vqc": round(ece_vec_vqc, 4),
            "poe": round(ece_vec_poe, 4),
            "gated": round(ece_vec_gat, 4)
        },
        "closes_calibration_asymmetry": bool(abs(ece_vec_vqc - ece_vec_poe) < 0.003)
    }
    print(f"Vector Scaling NLL: VQC={nll_vec_vqc:.5f}, PoE={nll_vec_poe:.5f}, dNLL={nll_vec_vqc - nll_vec_poe:+.5f}", flush=True)
    print(f"Vector Scaling ECE: VQC={ece_vec_vqc:.4f}, PoE={ece_vec_poe:.4f}, Gated={ece_vec_gat:.4f}", flush=True)
    
    summary = {
        "per_class_table": per_class_table,
        "aggregate_metrics": {
            "prevalence_weighted": {
                "vqc_vs_poe": {
                    "mean": round(float(np.mean(boot_prev_weighted_poe)), 5),
                    "ci95": [round(float(np.percentile(boot_prev_weighted_poe, 2.5)), 5),
                             round(float(np.percentile(boot_prev_weighted_poe, 97.5)), 5)]
                },
                "vqc_vs_gated": {
                    "mean": round(float(np.mean(boot_prev_weighted_gat)), 5),
                    "ci95": [round(float(np.percentile(boot_prev_weighted_gat, 2.5)), 5),
                             round(float(np.percentile(boot_prev_weighted_gat, 97.5)), 5)]
                }
            },
            "class_balanced_macro_mean": {
                "vqc_vs_poe": {
                    "mean": round(macro_mean_poe, 5),
                    "ci90": [round(x, 5) for x in macro_ci90_poe],
                    "ci95": [round(x, 5) for x in macro_ci95_poe],
                    "equivalent_at_0_010": macro_equiv_poe
                },
                "vqc_vs_gated": {
                    "mean": round(macro_mean_gat, 5),
                    "ci90": [round(x, 5) for x in macro_ci90_gat],
                    "ci95": [round(x, 5) for x in macro_ci95_gat],
                    "equivalent_at_0_010": macro_equiv_gat
                }
            },
            "worst_class": {
                "vqc_vs_poe": {
                    "class": per_class_table[worst_class_idx_poe]["diagnosis"],
                    "mean": round(float(obs_class_deltas_poe[worst_class_idx_poe]), 5),
                    "ci95": [round(float(np.percentile(boot_worst_poe, 2.5)), 5),
                             round(float(np.percentile(boot_worst_poe, 97.5)), 5)]
                },
                "vqc_vs_gated": {
                    "class": per_class_table[worst_class_idx_gat]["diagnosis"],
                    "mean": round(float(obs_class_deltas_gat[worst_class_idx_gat]), 5),
                    "ci95": [round(float(np.percentile(boot_worst_gat, 2.5)), 5),
                             round(float(np.percentile(boot_worst_gat, 97.5)), 5)]
                }
            }
        },
        "vector_scaling_ablation": vector_scaling_summary
    }
    
    out_file = OUT_DIR / "a3_class_analysis.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote results to {out_file}", flush=True)

if __name__ == "__main__":
    main()
