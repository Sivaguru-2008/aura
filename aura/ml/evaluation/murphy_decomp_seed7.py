"""SUPERSEDED AND WRONG. Use ``aura.ml.evaluation.murphy_decomp`` instead.

Retained only so that the defect is on the record; do not run this to produce a
published number. Two errors:

1. ``murphy_decomposition`` bins on ``probs.max(axis=1)``. The identity
   ``Brier = Reliability - Resolution + Uncertainty`` holds for groups of
   *identical forecast vectors*; binning on the maximum probability pools
   different forecast vectors into one group, so the terms do not reconstruct
   the score.
2. ``main`` computes ``delta_brier`` from ``actual_brier`` while printing
   reliability and resolution from the binned path. The two are not the same
   quantity. The function returns ``binned_brier`` and ``main`` never prints it,
   which is why the mismatch went unnoticed: the reported terms implied
   dBrier = dREL - dRES = -0.0169 while the printed dBrier was +0.0118.

The replacement groups by k-means on the simplex, reports the reconstruction
residual instead of assuming it away, runs across all ten seeds and both circuit
widths, and writes an artifact. See RESEARCH_PAPER_v4.md Appendix A.1.

Original docstring follows.

Compute the Murphy proper-score decomposition of the Brier score for seed 7.

Decomposes the Brier score into:
  Brier = Reliability (Calibration) - Resolution (Refinement) + Uncertainty
where a lower Reliability is better, and a higher Resolution is better.
"""
import numpy as np
import time

from aura.common.config import ARTIFACTS, ensure_dirs, get_settings
from aura.common.mathx import softmax
from aura.services.fusion.device import make_qnode
from aura.services.safety.calibration import fit_temperature
from aura.ml.evaluation.quantum_study import load_evidence, score
from aura.ml.training.train_fusion import train_quantum

N_QUBITS = 8
N_LAYERS = 3
EPOCHS = 30
SEED = 7

def _quantum_logits(theta, W, b, X, n_qubits, n_layers, entangler):
    circuit = make_qnode(n_qubits, n_layers, interface="numpy", entangler=entangler)
    z = np.stack([np.asarray(v) for v in circuit(np.asarray(X), theta)], axis=-1)
    return z @ W.T + b

def murphy_decomposition(probs, targets, bins=10):
    N, C = probs.shape
    y = np.zeros((N, C))
    y[np.arange(N), targets] = 1.0
    
    # overall class frequencies
    y_bar = y.mean(axis=0)
    uncertainty = np.sum(y_bar * (1 - y_bar))
    
    # bin the max probabilities (confidence)
    conf = probs.max(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    
    reliability = 0.0
    resolution = 0.0
    
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i+1])
        if not m.any():
            continue
        nm = m.sum()
        p_m = probs[m].mean(axis=0)
        y_m = y[m].mean(axis=0)
        
        reliability += nm * np.sum((p_m - y_m)**2)
        resolution += nm * np.sum((y_m - y_bar)**2)
        
    reliability /= N
    resolution /= N
    
    brier = np.mean(np.sum((probs - y)**2, axis=1))
    
    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "binned_brier": reliability - resolution + uncertainty,
        "actual_brier": brier
    }

def main():
    ensure_dirs()
    # Load cached evidence
    Xtr, ytr, Xcal, ycal, Xte, yte, source = load_evidence(n=900, seed=7)
    
    print("=== Training 8-qubit VQC models for Murphy decomposition ===")
    
    # 1. Ring (Entangled)
    print("Training Ring VQC...")
    t0 = time.time()
    theta_r, Wr, br = train_quantum(
        Xtr, ytr, N_QUBITS, N_LAYERS, epochs=EPOCHS, seed=SEED, entangler="ring"
    )
    cal_r = _quantum_logits(theta_r, Wr, br, Xcal, N_QUBITS, N_LAYERS, "ring")
    test_r = _quantum_logits(theta_r, Wr, br, Xte, N_QUBITS, N_LAYERS, "ring")
    T_r = fit_temperature(cal_r, ycal)
    probs_r = np.array([softmax(r / T_r) for r in test_r])
    print(f"Ring VQC trained in {time.time() - t0:.1f}s. Temperature: {T_r:.3f}")
    
    # 2. Product (None)
    print("Training Product VQC...")
    t0 = time.time()
    theta_p, Wp, bp = train_quantum(
        Xtr, ytr, N_QUBITS, N_LAYERS, epochs=EPOCHS, seed=SEED, entangler="none"
    )
    cal_p = _quantum_logits(theta_p, Wp, bp, Xcal, N_QUBITS, N_LAYERS, "none")
    test_p = _quantum_logits(theta_p, Wp, bp, Xte, N_QUBITS, N_LAYERS, "none")
    T_p = fit_temperature(cal_p, ycal)
    probs_p = np.array([softmax(r / T_p) for r in test_p])
    print(f"Product VQC trained in {time.time() - t0:.1f}s. Temperature: {T_p:.3f}")
    
    decomp_r = murphy_decomposition(probs_r, yte)
    decomp_p = murphy_decomposition(probs_p, yte)
    
    print("\n=== Murphy proper-score decomposition of the Brier score (seed 7) ===")
    print(f"Uncertainty (inherent label variance): {decomp_r['uncertainty']:.4f}")
    print("\nRing VQC (Entangled):")
    print(f"  Reliability (Calibration, lower is better): {decomp_r['reliability']:.4f}")
    print(f"  Resolution (Refinement, higher is better): {decomp_r['resolution']:.4f}")
    print(f"  Actual Brier Score                      : {decomp_r['actual_brier']:.4f}")
    
    print("\nProduct VQC (Unentangled):")
    print(f"  Reliability (Calibration, lower is better): {decomp_p['reliability']:.4f}")
    print(f"  Resolution (Refinement, higher is better): {decomp_p['resolution']:.4f}")
    print(f"  Actual Brier Score                      : {decomp_p['actual_brier']:.4f}")
    
    delta_reliability = decomp_r['reliability'] - decomp_p['reliability']
    delta_resolution = decomp_r['resolution'] - decomp_p['resolution']
    delta_brier = decomp_r['actual_brier'] - decomp_p['actual_brier']
    
    print("\n=== Entanglement Effect (Ring minus Product) ===")
    print(f"  Delta Reliability (Calibration): {delta_reliability:+.4f} (Negative favors entanglement)")
    print(f"  Delta Resolution (Refinement)  : {delta_resolution:+.4f} (Positive favors entanglement)")
    print(f"  Delta Brier Score              : {delta_brier:+.4f} (Negative favors entanglement)")
    
    if delta_reliability < 0 and delta_resolution < 0:
        print("\nSUCCESS: Entanglement mathematically improves calibration (reliability decreases)")
        print("         but degrades refinement (resolution decreases).")
        print(f"         Since |Delta Resolution| ({abs(delta_resolution):.4f}) > |Delta Reliability| ({abs(delta_reliability):.4f}),")
        print("         the loss in refinement outweighs the gain in calibration, leaving the final Brier score worse.")

if __name__ == "__main__":
    main()
