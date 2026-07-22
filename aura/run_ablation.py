import sys
import os
import json
import numpy as np
from pathlib import Path

# Ensure AURA package is in path
PKG_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG_ROOT))

from common.mathx import softmax
from schemas.clinical import DIAGNOSES
from services.fusion.quantum import QuantumFusion
from services.fusion.classical import ClassicalFusion
from services.fusion.learnable import LearnableFusion
from services.agent.active_diagnosis import ActiveDiagnosisAgent
from services.recommend.engine import RecommendEngine
from services.safety.calibration import fit_temperature, expected_calibration_error
from ml.training.dataset import build_real_evidence_dataset

class CalibratedModel:
    def __init__(self, model, temperature: float):
        self.model = model
        self.temperature = temperature
        self.backend = getattr(model, "backend", "fusion")
        self.n_qubits = getattr(model, "n_qubits", 8)
        self.n_layers = getattr(model, "n_layers", 3)
        if hasattr(model, "measurement_entropy"):
            self.measurement_entropy = model.measurement_entropy
        if hasattr(model, "probs"):
            self.probs = model.probs

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self.model.logits(x) / self.temperature

def main():
    print("=== Starting AURA Fusion Ablation ===")
    
    # Load models
    q = QuantumFusion.load()
    c = ClassicalFusion.load()
    l = LearnableFusion.load()
    
    # Load real evidence dataset (validate split has ~300-400 patient records)
    X, y = build_real_evidence_dataset(n=600, split="validate")
    if X is None or len(X) < 50:
        # Fallback to synthetic split if MIMIC is unavailable or empty
        from ml.training.dataset import make_splits, build_evidence_dataset
        print("MIMIC dataset not found or too small. Falling back to synthetic split...")
        samples_val, samples_cal, samples_te = make_splits(500, seed=7)
        Xva, yva = build_evidence_dataset(samples_cal)
        Xte, yte = build_evidence_dataset(samples_te)
        X = np.concatenate([Xva, Xte], axis=0)
        y = np.concatenate([yva, yte], axis=0)
    else:
        print(f"Loaded {len(X)} real patient evidence vectors from MIMIC-CXR.")

    # Split 50/50 into calibration and test
    n_total = len(X)
    n_cal = n_total // 2
    rng = np.random.default_rng(42)
    indices = rng.permutation(n_total)
    cal_idx = indices[:n_cal]
    te_idx = indices[n_cal:]
    
    X_cal, y_cal = X[cal_idx], y[cal_idx]
    X_te, y_te = X[te_idx], y[te_idx]
    
    print(f"Calibration size: {len(X_cal)}, Test size: {len(X_te)}")
    
    backends_to_test = {
        "quantum": q,
        "classical": c,
        "learnable": l,
    }
    
    results = {}
    
    for name, model in backends_to_test.items():
        if model is None:
            print(f"Backend '{name}' not found. Skipping.")
            continue
            
        print(f"Evaluating {name} backend...")
        
        # 1. Compute raw logits
        logits_cal = np.array([model.logits(x) for x in X_cal])
        logits_te = np.array([model.logits(x) for x in X_te])
        
        # 2. Fit temperature calibration parameter on calibration split
        T = fit_temperature(logits_cal, y_cal)
        
        # 3. Create calibrated wrapper
        cal_model = CalibratedModel(model, T)
        
        # 4. Evaluate raw model performance on test split
        probs_te = np.array([softmax(logits / T) for logits in logits_te])
        preds_te = probs_te.argmax(axis=1)
        
        accuracy = float((preds_te == y_te).mean())
        ece = float(expected_calibration_error(probs_te, y_te))
        nll = float(-np.mean(np.log(np.clip(probs_te[np.arange(len(y_te)), y_te], 1e-15, 1.0))))
        
        # 5. Run Active Diagnostic Agent simulation on test split
        agent = ActiveDiagnosisAgent(
            fusion_model=cal_model,
            entropy_target_bits=0.6,
            confidence=0.85,
            max_tests=3
        )
        
        steps_list = []
        abstain_count = 0
        correct_at_commit = 0
        total_committed = 0
        initial_entropies = []
        final_entropies = []
        
        for i, x in enumerate(X_te):
            trajectory = agent.diagnose(x)
            steps_list.append(len(trajectory.steps) - 1)
            initial_entropies.append(trajectory.initial_entropy)
            final_entropies.append(trajectory.final_entropy)
            if trajectory.status == "abstain":
                abstain_count += 1
            else:
                total_committed += 1
                pred_cls = DIAGNOSES.index(trajectory.final_diagnosis)
                if pred_cls == y_te[i]:
                    correct_at_commit += 1
                    
        commit_rate = total_committed / len(X_te)
        accuracy_at_commit = correct_at_commit / total_committed if total_committed > 0 else 0.0
        
        results[name] = {
            "calibration_temperature": round(T, 4),
            "model_accuracy": round(accuracy, 4),
            "model_ece": round(ece, 4),
            "model_nll": round(nll, 4),
            "agent_avg_steps": round(float(np.mean(steps_list)), 4),
            "agent_abstain_rate": round(float(abstain_count / len(X_te)), 4),
            "agent_commit_rate": round(float(commit_rate), 4),
            "agent_accuracy_at_commit": round(float(accuracy_at_commit), 4),
            "agent_avg_initial_entropy": round(float(np.mean(initial_entropies)), 4),
            "agent_avg_final_entropy": round(float(np.mean(final_entropies)), 4),
        }
        
    # Write artifacts/fusion_ablation.json
    artifacts_dir = PKG_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / "fusion_ablation.json"
    
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Results successfully saved to {out_path}")
    
    # Print markdown table
    print("\n| Metric | Quantum Fusion (VQC) | Classical Fusion (PoE) | Learnable Fusion (Gated) |")
    print("|---|---|---|---|")
    print(f"| Calibration Temp (T) | {results.get('quantum', {}).get('calibration_temperature', '-')} | {results.get('classical', {}).get('calibration_temperature', '-')} | {results.get('learnable', {}).get('calibration_temperature', '-')} |")
    print(f"| Model Accuracy | {results.get('quantum', {}).get('model_accuracy', '-'):.2%} | {results.get('classical', {}).get('model_accuracy', '-'):.2%} | {results.get('learnable', {}).get('model_accuracy', '-'):.2%} |")
    print(f"| Model ECE | {results.get('quantum', {}).get('model_ece', '-'):.4f} | {results.get('classical', {}).get('model_ece', '-'):.4f} | {results.get('learnable', {}).get('model_ece', '-'):.4f} |")
    print(f"| Model NLL | {results.get('quantum', {}).get('model_nll', '-'):.4f} | {results.get('classical', {}).get('model_nll', '-'):.4f} | {results.get('learnable', {}).get('model_nll', '-'):.4f} |")
    print(f"| Avg Agent Steps | {results.get('quantum', {}).get('agent_avg_steps', '-'):.2f} | {results.get('classical', {}).get('agent_avg_steps', '-'):.2f} | {results.get('learnable', {}).get('agent_avg_steps', '-'):.2f} |")
    print(f"| Agent Abstain Rate | {results.get('quantum', {}).get('agent_abstain_rate', '-'):.2%} | {results.get('classical', {}).get('agent_abstain_rate', '-'):.2%} | {results.get('learnable', {}).get('agent_abstain_rate', '-'):.2%} |")
    print(f"| Agent Commit Rate | {results.get('quantum', {}).get('agent_commit_rate', '-'):.2%} | {results.get('classical', {}).get('agent_commit_rate', '-'):.2%} | {results.get('learnable', {}).get('agent_commit_rate', '-'):.2%} |")
    print(f"| Agent Acc @ Commit | {results.get('quantum', {}).get('agent_accuracy_at_commit', '-'):.2%} | {results.get('classical', {}).get('agent_accuracy_at_commit', '-'):.2%} | {results.get('learnable', {}).get('agent_accuracy_at_commit', '-'):.2%} |")
    print(f"| Avg Init Entropy | {results.get('quantum', {}).get('agent_avg_initial_entropy', '-'):.3f} (Qubit) | {results.get('classical', {}).get('agent_avg_initial_entropy', '-'):.3f} (Dx) | {results.get('learnable', {}).get('agent_avg_initial_entropy', '-'):.3f} (Dx) |")
    print(f"| Avg Final Entropy | {results.get('quantum', {}).get('agent_avg_final_entropy', '-'):.3f} (Qubit) | {results.get('classical', {}).get('agent_avg_final_entropy', '-'):.3f} (Dx) | {results.get('learnable', {}).get('agent_avg_final_entropy', '-'):.3f} (Dx) |")
    print("\n=== Ablation Completed successfully ===")

if __name__ == "__main__":
    main()
