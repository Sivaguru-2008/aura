import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path("e:/AURA/aura-main")
RES_DIR = ROOT / "results" / "revision_r4"
BENCH_DIR = ROOT / "bench" / "revision_r4"

GIT_HASH = "a54ca46f2b7a05bda37fc2a67d9045241336a2ed"
TIMESTAMP = datetime.now(timezone.utc).isoformat()
SEEDS = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]

ARTIFACTS = {
    "classical_seeds_summary.json": {
        "script": "bench/revision_r4/run_a1_classical_seeds.py",
        "description": "10-seed evaluation of C2 tuned PoE, C2 tuned Gated Head, and C1 default PoE on N=56,534 held-out patients."
    },
    "a1_quantum_seeds_summary.json": {
        "script": "bench/revision_r4/run_a1_quantum_seeds.py",
        "description": "10-seed evaluation of C1 default VQC and C2 tuned VQC, paired t-tests, seed-level TOST, and hierarchical bootstrap."
    },
    "a2_selection_robustness.json": {
        "script": "bench/revision_r4/run_a2_selection_robustness.py",
        "description": "36-config selection stability across 20 resampled dev splits, plus full 60 classical grid sensitivity analysis."
    },
    "a3_class_analysis.json": {
        "script": "bench/revision_r4/run_a3_class_analysis.py",
        "description": "Class-conditional DeltaNLL with 95% bootstrap CIs, macro-mean TOST failure, and vector scaling ablation."
    },
    "domain_shift_analysis.json": {
        "script": "bench/revision_r4/run_a4_domain_shift.py",
        "description": "Institutional transfer from MIMIC-CXR to CheXpert, angle-wrapping verification, and Fourier trigonometric baseline comparison."
    },
    "m1_effective_order.json": {
        "script": "bench/revision_r4/run_a5_m1_order.py",
        "description": "Interaction-order energy spectrum excluding constant Z8 reference qubit across 10 C1b seeds and confirmatory circuit."
    },
    "m2_completeness.json": {
        "script": "bench/revision_r4/run_a6_m2_surrogates.py",
        "description": "Classical order-1 and order-2 trigonometric Fourier surrogates and polynomial linear regression on N=56,534 test patients."
    },
    "ckd_protocol_grid.json": {
        "script": "bench/revision_r4/run_a7_ckd_grid.py",
        "description": "90-protocol grid for Chronic Kidney Disease literature replication across scalers, cross-validation schemes, and models."
    }
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()

def build_manifest():
    entries = {}
    for fname, meta in ARTIFACTS.items():
        src = BENCH_DIR / fname
        dst = RES_DIR / fname
        if src.exists():
            if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
                dst.write_bytes(src.read_bytes())
            entries[fname] = {
                "script": meta["script"],
                "description": meta["description"],
                "sha256": sha256(dst),
                "bytes": dst.stat().st_size
            }
        elif dst.exists():
            entries[fname] = {
                "script": meta["script"],
                "description": meta["description"],
                "sha256": sha256(dst),
                "bytes": dst.stat().st_size
            }
        else:
            print(f"Warning: {fname} does not exist yet.")

    manifest = {
        "title": "Revision R4 Master Experimental Artifacts Manifest",
        "target_journal": "npj Digital Medicine",
        "git_branch": "revision-r4",
        "git_hash": GIT_HASH,
        "seed_list": SEEDS,
        "timestamp_utc": TIMESTAMP,
        "artifacts": entries
    }

    out = RES_DIR / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"Successfully generated manifest at {out} with {len(entries)} artifacts.")

if __name__ == "__main__":
    build_manifest()
