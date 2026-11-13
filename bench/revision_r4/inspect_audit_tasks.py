import json
from pathlib import Path
import numpy as np

FA_DIR = Path("e:/AURA/aura-main/bench/field_audit/results")

for p in FA_DIR.glob("*.json"):
    try:
        d = json.loads(p.read_text())
        print(f"=== {p.name} ===")
        if "study" in d:
            print("  study:", d.get("study"), "metric:", d.get("primary_metric"))
        if "reproduction" in d:
            print("  reproduction keys:", list(d["reproduction"].keys()) if isinstance(d["reproduction"], dict) else d["reproduction"])
        if "symmetric" in d:
            sym = d["symmetric"]
            if isinstance(sym, dict):
                print("  symmetric verdict:", sym.get("verdict", {}).get("verdict") if "verdict" in sym else sym.get("verdict"))
                if "full" in sym:
                    print("  full verdict:", sym["full"].get("verdict"))
    except Exception as e:
        print(f"Error reading {p}: {e}")
