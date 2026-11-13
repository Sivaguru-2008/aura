import itertools
import json
from pathlib import Path
import numpy as np

from aura.common.config import ARTIFACTS
from aura.ml.evaluation import v6_confirmatory as v6
from aura.ml.evaluation.v6_mechanism import (
    _P, _unitary, _pauli_coeffs, _features, _readout
)

PART = ARTIFACTS / "v6"
OUT_DIR = Path(__file__).resolve().parent

def effective_order_mask(n: int, informative_qubits: int = 7) -> np.ndarray:
    """Order mask counting only the first informative_qubits."""
    idx = np.array(list(itertools.product(range(3), repeat=n)))
    # Only count non-identity in the first informative_qubits
    eff = (idx[:, :informative_qubits] > 0).sum(1)
    return eff.reshape((3,) * n)

def spectrum_effective(theta, W, b, n, entangler, X, informative_qubits=7) -> dict:
    U = _unitary(theta, n, entangler)
    eff_order = effective_order_mask(n, informative_qubits)
    raw_order = (np.array(list(itertools.product(range(3), repeat=n))) > 0).sum(1).reshape((3,) * n)
    
    coeffs = []
    for j in range(n):
        Zj = np.diag([1.0 if ((k >> (n - 1 - j)) & 1) == 0 else -1.0 for k in range(2 ** n)])
        O = U.conj().T @ Zj @ U
        coeffs.append(_pauli_coeffs(O, n))
    coeffs = np.stack(coeffs)  # [n_readout, 3,...,3]
    
    # Energy by raw order
    raw_energy = {int(o): float((coeffs[:, raw_order == o] ** 2).sum()) for o in range(n + 1)}
    raw_tot = sum(raw_energy.values())
    
    # Energy by effective order (0..informative_qubits)
    eff_energy = {int(o): float((coeffs[:, eff_order == o] ** 2).sum()) for o in range(informative_qubits + 1)}
    eff_tot = sum(eff_energy.values())
    
    # Logit variance share by effective order
    # Logit decomposition on data: logit_k = sum_j W_kj <Z_j> + b_k
    flat = coeffs.reshape(n, -1)  # [n, 3^n]
    lin = W @ flat               # [K, 3^n]
    ord_flat = eff_order.ravel()
    low = np.where(ord_flat <= 1)[0]
    
    r2 = []
    for i in range(0, len(X), 4000):
        F, _ = _features(X[i:i + 4000], n)
        full = F @ lin.T
        trunc = F[:, low] @ lin[:, low].T
        r2.append((full, trunc))
    full = np.concatenate([a for a, _ in r2])
    trunc = np.concatenate([b_ for _, b_ in r2])
    var_full = full.var(0)
    resid = (full - trunc).var(0)
    eff_share_ge2 = float(np.mean(np.where(var_full > 1e-12, resid / np.maximum(var_full, 1e-12), 0.0)))
    
    return {
        "qubits": n,
        "informative_qubits": informative_qubits,
        "entangler": entangler,
        "effective_coefficient_energy_share_by_order": {
            o: round(e / eff_tot, 6) for o, e in eff_energy.items()
        },
        "effective_energy_share_ge2": round(sum(eff_energy[o] for o in range(2, informative_qubits + 1)) / eff_tot, 6),
        "raw_energy_share_ge2": round(sum(raw_energy[o] for o in range(2, n + 1)) / raw_tot, 6),
        "effective_logit_variance_share_order_ge2": round(eff_share_ge2, 6),
    }

def main():
    Xte = v6.test_set()[0]
    results = {}
    
    # 1. C1 default quantum circuit (8 qubits, ring, L=3, s=1.0)
    z_c1 = np.load(PART / "C1_y_u0_quantum_params.npz")
    results["C1_default_vqc"] = spectrum_effective(z_c1["p0"], z_c1["p1"], z_c1["p2"], 8, "ring", Xte, 7)
    print("C1 Default VQC:", json.dumps(results["C1_default_vqc"], indent=2))
    
    # 2. C2 winner quantum circuit (8 qubits, ring, L=4, s=0.5, sd=0.01)
    # The winner file from C2_quantum_grid.json is C2_quantum_20.npz
    c2_grid = json.loads((PART / "C2_quantum_grid.json").read_text())
    c2_winner_rec = min(c2_grid, key=lambda r: r["best_calibration_nll"])
    z_c2 = np.load(PART / c2_winner_rec["file"])
    results["C2_winner_vqc"] = spectrum_effective(z_c2["p0"], z_c2["p1"], z_c2["p2"], 8, "ring", Xte, 7)
    print("C2 Winner VQC:", json.dumps(results["C2_winner_vqc"], indent=2))
    
    # 3. C1b 7-qubit circuits (10 seeds, ring and none)
    from aura.ml.evaluation.revision_v51 import SEEDS
    c1b_rows = []
    for s in SEEDS:
        r = {"seed": s}
        for ent in ("ring", "none"):
            p = PART / f"C1b_seed{s}_{ent}_params.npz"
            if p.exists():
                q = np.load(p)
                # In C1b, all 7 qubits are informative
                r[ent] = spectrum_effective(q["theta"], q["W"], q["b"], 7, ent, Xte, 7)
        c1b_rows.append(r)
    results["C1b_7q"] = c1b_rows
    
    # Summary of C1b
    ring_eff_ge2 = [r["ring"]["effective_logit_variance_share_order_ge2"] for r in c1b_rows if "ring" in r]
    none_eff_ge2 = [r["none"]["effective_logit_variance_share_order_ge2"] for r in c1b_rows if "none" in r]
    ring_energy_ge2 = [r["ring"]["effective_energy_share_ge2"] for r in c1b_rows if "ring" in r]
    none_energy_ge2 = [r["none"]["effective_energy_share_ge2"] for r in c1b_rows if "none" in r]
    
    results["C1b_summary"] = {
        "ring_eff_logit_var_ge2_mean": round(float(np.mean(ring_eff_ge2)), 6),
        "ring_eff_logit_var_ge2_sd": round(float(np.std(ring_eff_ge2, ddof=1)), 6),
        "none_eff_logit_var_ge2_mean": round(float(np.mean(none_eff_ge2)), 6),
        "ring_energy_ge2_mean": round(float(np.mean(ring_energy_ge2)), 6),
        "none_energy_ge2_mean": round(float(np.mean(none_energy_ge2)), 6),
    }
    print("C1b Summary:", json.dumps(results["C1b_summary"], indent=2))
    
    out_file = OUT_DIR / "m1_effective_order.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_file}")

if __name__ == "__main__":
    main()
