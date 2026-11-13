import json
from pathlib import Path
import numpy as np

FA_DIR = Path("e:/AURA/aura-main/bench/field_audit/results")

def inspect_task(name, path):
    if not path.exists():
        print(f"File not found: {path}")
        return
    d = json.loads(path.read_text())
    print(f"\n=================== {name} ===================")
    print("Reported:", d.get("reported"))
    print("Primary metric:", d.get("primary_metric") or d.get("metric"))
    
    # Untuned reproduction comparison
    if "reproduction" in d:
        print("\n--- Untuned / Paper Reproduction ---")
        rep = d["reproduction"]
        if isinstance(rep, dict):
            for k, v in rep.items():
                if k != "protocol":
                    print(f"  {k}: {v}")
    
    # Symmetric / Tuned comparison
    if "symmetric" in d:
        print("\n--- Symmetric / Tuned Comparison ---")
        sym = d["symmetric"]
        if "full" in sym:
            f = sym["full"]
            v = f.get("verdict", {})
            diff = v.get("difference_quantum_minus_classical", {})
            print(f"  Delta (Q - C): {diff.get('mean')} (CI: {diff.get('ci95')}, p: {diff.get('p')})")
            print(f"  Verdict: {v.get('verdict')}")
            print(f"  Arms: Q={v.get('arm_means_sd', {}).get('quantum')}, C={v.get('arm_means_sd', {}).get('classical')}")
        elif "verdict" in sym:
            v = sym["verdict"]
            diff = v.get("difference_quantum_minus_classical", {})
            print(f"  Delta (Q - C): {diff.get('mean')} (CI: {diff.get('ci95')}, p: {diff.get('p')})")
            print(f"  Verdict: {v.get('verdict')}")

def main():
    inspect_task("CKD", FA_DIR / "ckd.json")
    inspect_task("Parkinsons", FA_DIR / "stack_parkinsons.json")
    inspect_task("Pima", FA_DIR / "stack_pima.json")
    inspect_task("EEG", FA_DIR / "eeg_qsvm.json")
    inspect_task("Glioma", FA_DIR / "glioma.json")
    
    # Factors files
    if (FA_DIR / "factors_parkinsons.json").exists():
        fp = json.loads((FA_DIR / "factors_parkinsons.json").read_text())
        print("\n=== Factors Parkinsons Cells ===")
        for c in fp["cells"]:
            print(f"  pca={c['pca_scope']:8s} unit={c['cv_unit']:9s} comp={c['comparator']:15s} q={c['quantum_mean']:.3f} c={c['comparator_mean']:.3f} d={c['delta']['mean']:+.3f} ci={c['delta']['ci95']} p={c['delta']['p']:.4f}")

if __name__ == "__main__":
    main()
