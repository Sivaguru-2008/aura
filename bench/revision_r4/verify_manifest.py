"""Level A verification script for Revision R4.
Verifies that all artifacts listed in results/revision_r4/manifest.json exist and match their SHA-256 hashes.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "results" / "revision_r4" / "manifest.json"

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()

def verify():
    if not MANIFEST_PATH.exists():
        print(f"ERROR: Manifest {MANIFEST_PATH} not found.")
        sys.exit(1)
        
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", {})
    if not artifacts:
        print("ERROR: Manifest has no artifacts recorded.")
        sys.exit(1)
        
    res_dir = ROOT / "results" / "revision_r4"
    failed = []
    
    print(f"Verifying {len(artifacts)} Revision R4 artifacts against pinned SHA-256...")
    for fname, meta in artifacts.items():
        fpath = res_dir / fname
        if not fpath.exists():
            failed.append(f"MISSING: {fname}")
            continue
        actual_hash = sha256(fpath)
        expected_hash = meta["sha256"]
        if actual_hash != expected_hash:
            failed.append(f"HASH MISMATCH: {fname} (expected {expected_hash[:12]}..., got {actual_hash[:12]}...)")
        else:
            print(f"  [OK] {fname} ({meta['bytes']} bytes, SHA-256: {actual_hash[:12]}...)")
            
    if failed:
        print("\nLevel A Verification FAILED:")
        for err in failed:
            print(f"  {err}")
        sys.exit(1)
        
    print(f"\nLevel A Verification PASSED: All {len(artifacts)} artifacts intact and bit-for-bit verified.")
    return 0

if __name__ == "__main__":
    sys.exit(verify())
