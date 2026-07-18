import sys
import os
import json
from pathlib import Path

# Setup sys.path to load AURA modules
sys.path.append(os.path.join(os.path.dirname(__file__), "aura"))

# Configure backend environment variables first so settings load properly
os.environ["AURA_VISION_BACKEND"] = "timm"
os.environ["AURA_VISION_ARCH"] = "densenet121"

from services.vision.io import study_from_cxr
from gateway.pipeline import Pipeline

async def main():
    if len(sys.argv) < 2:
        print("Usage: python test_image.py <path_to_xray_image>")
        sys.exit(1)
        
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"Error: File not found at {img_path}")
        sys.exit(1)
        
    print(f"Loading image {img_path} and setting up pipeline...")
    
    # Build study input from the real CXR image
    study = study_from_cxr(img_path)
    
    # Run the full AURA reasoning pipeline
    pipeline = Pipeline()
    print("Running AURA clinical copilot pipeline...")
    bundle = await pipeline.run(study, case_id="CASE-TEST-REAL")
    
    # Format and display the outputs
    print("\n" + "="*50)
    print("                AURA REPORT")
    print("="*50)
    
    print("\n[1] Vision Model Findings:")
    if bundle.vision:
        for f in bundle.vision.findings:
            print(f"  * {f.finding:<20}: {f.probability * 100:.2f}%")
        
    print("\n[2] Safety Calibrated Conformal Set:")
    if bundle.safety:
        if bundle.safety.abstained:
            print(f"  * Status: ABSTAINED (Reason: {bundle.safety.abstention_reason.value})")
        else:
            print(f"  * Status: PASSED (Top diagnosis: {bundle.safety.top.value})")
            print(f"  * 90% Confidence Set: {', '.join([d.value for d in bundle.safety.conformal_set])}")
        
    print("\n[3] Calibrated Diagnosis Differential (Quantum Posterior):")
    if bundle.fusion:
        for dx, prob in sorted(bundle.fusion.posterior.items(), key=lambda kv: -kv[1]):
            std_val = bundle.fusion.posterior_std.get(dx, 0.0)
            print(f"  * {dx.value:<20}: {prob * 100:.2f}% ± {std_val * 100:.2f}%")
        
    print("\n[4] Explainability (Evidence Attributions & Counterfactuals):")
    if bundle.explanation:
        for node, attr in sorted(bundle.explanation.evidence_attribution.items(), key=lambda kv: -abs(kv[1])):
            cf = bundle.explanation.counterfactuals.get(node, 0.0)
            print(f"  * {node:<20}: Attribution {attr:+.4f} (Counterfactual: {cf:+.4f})")
        
    print("\n[5] Next Best Evidence Recommendation:")
    for i, r in enumerate(bundle.recommendations[:3]):
        print(f"  * Rank {i+1}: {r.display:<20} (EIG: {r.expected_info_gain:.4f}, Cost: {r.cost_tier})")
        
    print("\n" + "="*50)
    print("Grounded Clinical Narrative:")
    print("="*50)
    if bundle.report:
        print(f"Findings: {bundle.report.findings_text}")
        print(f"Impression: {bundle.report.impression_text}")
        print(f"Recommendation: {bundle.report.recommendation_text}")
    print("="*50 + "\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
