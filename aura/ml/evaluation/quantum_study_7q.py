"""7-Qubit entanglement ablation study across 10 training seeds.

This script drops the dead eighth channel (prior_risk) to evaluate the variational
quantum circuit at 7 qubits, eliminating the coupling confound to a constant register
(CRITICAL-1).
"""
import json
import time
from pathlib import Path
import numpy as np
import scipy.stats as stats
import torch

from aura.common.config import ARTIFACTS, ensure_dirs, get_settings
from aura.common.mathx import softmax
from aura.services.fusion.device import make_qnode
from aura.services.safety.calibration import (
    expected_calibration_error,
    fit_temperature,
)
from aura.ml.evaluation.quantum_study import load_evidence, score

OUTPUT = ARTIFACTS / "entanglement_multiseed_7q.json"
SEEDS = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
EPOCHS = 30
N_QUBITS = 7
N_LAYERS = 3

def _quantum_logits(theta, W, b, X, n_qubits, n_layers, entangler):
    circuit = make_qnode(n_qubits, n_layers, interface="numpy", entangler=entangler)
    z = np.stack([np.asarray(v) for v in circuit(np.asarray(X), theta)], axis=-1)
    return z @ W.T + b

def main():
    ensure_dirs()
    s = get_settings()
    
    # Load cached evidence (which has 8 channels) and slice to first 7 channels
    Xtr_8, ytr, Xcal_8, ycal, Xte_8, yte, source = load_evidence(n=900, seed=7)
    Xtr = Xtr_8[:, :N_QUBITS]
    Xcal = Xcal_8[:, :N_QUBITS]
    Xte = Xte_8[:, :N_QUBITS]
    
    from aura.ml.training.train_fusion import train_quantum
    
    per_seed = []
    started = time.time()
    
    print(f"=== Running 7-Qubit Ablation Study (10 seeds, {EPOCHS} epochs) ===")
    print(f"Data source: {source} | train={len(ytr)} cal={len(ycal)} test={len(yte)}")
    
    for idx, seed in enumerate(SEEDS, 1):
        print(f"\n[Seed {idx}/10] seed={seed} ...")
        
        # 1. Train Ring (Entangled)
        t0 = time.time()
        theta_r, Wr, br = train_quantum(
            Xtr, ytr, N_QUBITS, N_LAYERS, epochs=EPOCHS, seed=seed, entangler="ring"
        )
        cal_r = _quantum_logits(theta_r, Wr, br, Xcal, N_QUBITS, N_LAYERS, "ring")
        test_r = _quantum_logits(theta_r, Wr, br, Xte, N_QUBITS, N_LAYERS, "ring")
        m_r = score(cal_r, ycal, test_r, yte)
        t_r = time.time() - t0
        
        # 2. Train Product (None)
        t0 = time.time()
        theta_p, Wp, bp = train_quantum(
            Xtr, ytr, N_QUBITS, N_LAYERS, epochs=EPOCHS, seed=seed, entangler="none"
        )
        cal_p = _quantum_logits(theta_p, Wp, bp, Xcal, N_QUBITS, N_LAYERS, "none")
        test_p = _quantum_logits(theta_p, Wp, bp, Xte, N_QUBITS, N_LAYERS, "none")
        m_p = score(cal_p, ycal, test_p, yte)
        t_p = time.time() - t0
        
        delta_nll = m_r["nll"] - m_p["nll"]
        delta_acc = m_r["accuracy"] - m_p["accuracy"]
        delta_ece = m_r["ece"] - m_p["ece"]
        
        print(f"  Ring: acc={m_r['accuracy']:.4f} nll={m_r['nll']:.4f} ece={m_r['ece']:.4f} in {t_r:.1f}s")
        print(f"  Prod: acc={m_p['accuracy']:.4f} nll={m_p['nll']:.4f} ece={m_p['ece']:.4f} in {t_p:.1f}s")
        print(f"  Delta NLL: {delta_nll:+.4f} | Acc: {delta_acc:+.4f} | ECE: {delta_ece:+.4f}")
        
        per_seed.append({
            "seed": seed,
            "entangled": {
                "accuracy": m_r["accuracy"],
                "nll": m_r["nll"],
                "ece": m_r["ece"],
                "temperature": m_r["temperature"],
                "mean_confidence": m_r["mean_confidence"],
                "train_seconds": round(t_r, 2)
            },
            "product": {
                "accuracy": m_p["accuracy"],
                "nll": m_p["nll"],
                "ece": m_p["ece"],
                "temperature": m_p["temperature"],
                "mean_confidence": m_p["mean_confidence"],
                "train_seconds": round(t_p, 2)
            },
            "delta_nll": round(delta_nll, 4),
            "delta_accuracy": round(delta_acc, 4),
            "delta_ece": round(delta_ece, 4)
        })
        
    # Analyze results
    analysis = {}
    metrics_keys = ["delta_nll", "delta_accuracy", "delta_ece"]
    raw_p_values = {}
    
    for key in metrics_keys:
        vals = [s[key] for s in per_seed]
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1))
        
        # t-test (one-sample against zero)
        t_stat, t_p = stats.ttest_1samp(vals, 0.0)
        
        # Wilcoxon signed-rank
        try:
            w_stat, w_p = stats.wilcoxon(vals, alternative="two-sided")
        except Exception:
            w_stat, w_p = float("nan"), float("nan")
            
        raw_p_values[key] = {
            "vals": vals,
            "mean": mean,
            "sd": sd,
            "t_p": t_p,
            "wilcoxon_p": w_p
        }

    # Holm-Bonferroni correction (m=3 metrics)
    sorted_keys = sorted(metrics_keys, key=lambda k: raw_p_values[k]["wilcoxon_p"])
    m_tests = len(metrics_keys)
    
    analysis = {
        "purpose": "Across-seed replication of the 7-qubit entanglement ablation. dropping dead prior_risk channel.",
        "protocol": {
            "seeds": SEEDS,
            "epochs": EPOCHS,
            "n_qubits": N_QUBITS,
            "n_layers": N_LAYERS,
            "data": "cached quantum_study_evidence.npz (first 7 channels)",
            "n_train": len(ytr),
            "n_cal": len(ycal),
            "n_test": len(yte)
        },
        "per_seed": per_seed
    }
    
    for rank, key in enumerate(sorted_keys, 1):
        raw = raw_p_values[key]
        corrected_w_p = min(1.0, raw["wilcoxon_p"] * (m_tests - rank + 1))
        corrected_t_p = min(1.0, raw["t_p"] * (m_tests - rank + 1))
        
        # calculate CI of the mean
        n = len(raw["vals"])
        se = raw["sd"] / np.sqrt(n)
        ci = stats.t.interval(0.95, n-1, loc=raw["mean"], scale=se)
        
        # seeds favoring ring
        if key == "delta_accuracy":
            fav = sum(1 for s in per_seed if s["delta_accuracy"] > 0)
        elif key == "delta_nll":
            fav = sum(1 for s in per_seed if s["delta_nll"] < 0)
        else: # delta_ece
            fav = sum(1 for s in per_seed if s["delta_ece"] < 0)
            
        analysis[key] = {
            "per_seed": raw["vals"],
            "mean": round(raw["mean"], 4),
            "sd_across_seeds": round(raw["sd"], 4),
            "ci95_mean": [round(ci[0], 4), round(ci[1], 4)],
            "excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
            "t_p_two_sided": round(float(raw["t_p"]), 4),
            "t_p_holm": round(float(corrected_t_p), 4),
            "wilcoxon_p": round(float(raw["wilcoxon_p"]), 4),
            "wilcoxon_p_holm": round(float(corrected_w_p), 4),
            "n_seeds_favouring_entanglement": fav,
        }

    OUTPUT.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"\n[done] wrote results to {OUTPUT} in {time.time() - started:.1f}s")
    
    # Print markdown table
    print("\n| Metric | Mean Delta | SD | 95% CI | t p | Wilcoxon p | Holm p | Seeds Favouring Ring |")
    print("|---|---:|---:|---|---:|---:|---:|---:|")
    for key, name in [("delta_nll", "Negative log-likelihood"),
                      ("delta_accuracy", "Accuracy"),
                      ("delta_ece", "Expected calibration error")]:
        a = analysis[key]
        sign = "+" if a["mean"] > 0 else ""
        print(f"| {name} | {sign}{a['mean']:.4f} | {a['sd_across_seeds']:.4f} | {sign}{a['ci95_mean'][0]:.4f} to {a['ci95_mean'][1]:.4f} | {a['t_p_two_sided']:.4f} | {a['wilcoxon_p']:.4f} | {a['wilcoxon_p_holm']:.4f} | {a['n_seeds_favouring_entanglement']} of 10 |")

if __name__ == "__main__":
    main()
