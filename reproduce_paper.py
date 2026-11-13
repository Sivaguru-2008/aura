"""One command to reproduce the manuscript's numbers.

    python reproduce_paper.py            # quick: verify inputs, rebuild figures, run the drift tests
    python reproduce_paper.py --full     # also re-run every analysis from cached evidence (hours)
    python reproduce_paper.py --list     # print each step and the artifact it writes

Data are not redistributed (MIMIC-CXR and CheXpert require credentialed access). The quick
mode needs only the released artifacts. The full mode needs the cached evidence tensors,
which are regenerated from the images by the scripts named in each step and require
``AURA_MIMIC_ROOT`` and the CheXpert-v1.0-small path in ``chexpert_external.CHEXPERT_ROOT``.

Every step is a subprocess, so a failure names the step and stops the run; nothing is
skipped silently.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
M = "aura.ml.evaluation."

QUICK = [
    ("Input provenance: pinned SHA-256 of the fusion evidence and predictions",
     [PY, "-m", M + "provenance"], "evidence_ledger.json"),
    ("Figure 1: forest plot (every evaluation choice; published claims re-tested)",
     [PY, "-m", M + "v6_figures"], "docs/figures/fig_v6_forest.png, v6_forest.json"),
    ("Figure 2: mechanism (interaction-order spectrum; kernel bandwidth sweep)",
     [PY, "-m", M + "v6_figures", "mechanism"], "docs/figures/fig_v6_mechanism.png"),
    ("Drift tests: every manuscript number equals its artifact",
     [PY, "-m", "pytest", "-q", "aura/tests/test_manuscript_numbers.py",
      "aura/tests/test_doc_numbers.py"], "(test report)"),
    ("Revision R4: Level A integrity verification of all new artifacts in results/revision_r4/manifest.json",
     [PY, "bench/revision_r4/verify_manifest.py"], "results/revision_r4/manifest.json"),
]

FULL = [
    ("Matched-optimiser comparison, MIMIC-CXR n=447 (§ Results, Fig 1A)",
     [PY, "-m", M + "matched_optimiser"], "fusion_matched_optimiser.json"),
    ("Positive control T1/T2a/T2b", [PY, "-m", M + "nonlinear_control"],
     "nonlinear_control.json"),
    ("Revision analyses: TOST, patient bootstrap, re-splits, re-uploading",
     [PY, "-m", M + "revision_v51", "summarize"], "revision_v51.json"),
    ("CheXpert frozen transfer (pre-registered)", [PY, "-m", M + "chexpert_external"],
     "chexpert_external.json"),
    ("CheXpert out-of-sample evidence E1", [PY, "-m", M + "chexpert_internal", "e1"],
     "chexpert_internal/E1_*.json"),
    ("CheXpert out-of-sample evidence E2", [PY, "-m", M + "chexpert_internal", "e2"],
     "chexpert_internal/E2_*.json"),
    ("CheXpert out-of-sample summary", [PY, "-m", M + "chexpert_internal", "summarize"],
     "chexpert_internal.json"),
    ("v6 simulator verification", [PY, "-m", M + "v6_confirmatory", "simcheck"],
     "v6/simcheck.json"),
    ("v6 C1 primary arms (labels: uncertain = negative)",
     [PY, "-m", M + "v6_confirmatory", "c1", "y_u0"], "v6/C1_y_u0_*"),
    ("v6 C1 sensitivity arms (uncertain = positive)",
     [PY, "-m", M + "v6_confirmatory", "c1", "y_u1"], "v6/C1_y_u1_*"),
    ("v6 C1b entanglement ablation", [PY, "-m", M + "v6_confirmatory", "c1b"],
     "v6/C1b_*"),
    ("v6 C2 symmetric search: classical", [PY, "-m", M + "v6_confirmatory", "c2", "classical"],
     "v6/C2_classical_*"),
    ("v6 C2 symmetric search: gated", [PY, "-m", M + "v6_confirmatory", "c2", "learnable"],
     "v6/C2_learnable_*"),
    ("v6 C2 symmetric search: quantum", [PY, "-m", M + "v6_confirmatory", "c2", "quantum"],
     "v6/C2_quantum_*"),
    ("v6 scoring on untouched patients + pre-registered reading",
     [PY, "-m", M + "v6_confirmatory", "summarize"], "v6_confirmatory.json"),
    ("v6 mechanism M1-M3", [PY, "-c",
                            "import subprocess,sys;[subprocess.check_call([sys.executable,'-m',"
                            "'aura.ml.evaluation.v6_mechanism',k]) for k in ('m1','m2','m3')]"],
     "v6_mechanism.json"),
    ("Field audit: re-test of published claims",
     [PY, "-m", "bench.field_audit.studies", "all"], "bench/field_audit/results/*.json"),
    ("Field audit: factor separation F1-F3", [PY, "-c",
     "import subprocess,sys;[subprocess.check_call([sys.executable,'-m',"
     "'bench.field_audit.factors',k]) for k in ('parkinsons','pima','eeg')]"],
     "bench/field_audit/results/factors_*.json"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    steps = (FULL + QUICK) if args.full else QUICK
    if args.list:
        for i, (name, cmd, out) in enumerate(steps, 1):
            print(f"{i:2d}. {name}\n    -> {out}\n    $ {' '.join(cmd[1:])}")
        return 0
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    t0 = time.time()
    for i, (name, cmd, out) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {name}", flush=True)
        r = subprocess.run(cmd, cwd=ROOT, env=env)
        if r.returncode != 0:
            print(f"\nFAILED at step {i}: {name} (exit {r.returncode})")
            return r.returncode
    print(f"\nAll {len(steps)} steps passed in {time.time() - t0:.0f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
