"""Test that a grade column is attached to the right images, from the images alone.

Why this exists
---------------
Grades reach the kernel comparison through a join: the HDF5 redistribution names
subjects ``volume_N``, the reader maps that to ``BraTS20_Training_{N:03d}``, and the
release's ``name_mapping.csv`` gives the grade for that name. The 2026-09-13 audit found
one corpus on disk (``datasets/brats2021_hdf5``) whose join was wrong. Checking names
against names cannot catch that — both columns can look perfect while pointing at
different patients. This checks names against pixels.

The test, fixed before either corpus was scored
-----------------------------------------------
Contrast enhancement is a defining radiological difference between the grades: most
high-grade gliomas enhance, and a large fraction of low-grade gliomas do not. So:

* **Primary.** "Non-enhancing" = zero voxels of the enhancing-tumour label in the
  expert segmentation. Under a correct join, the non-enhancing proportion is higher in
  subjects labelled LGG than in subjects labelled HGG (one-sided Fisher exact test,
  alpha 0.01). Under a join unrelated to the images, the two proportions are equal in
  expectation.
* **Secondary.** AUROC of the enhancing fraction of tumour volume for separating the
  labelled grades, with a label-permutation null (10,000 permutations).

Both corpora are run through the same function. The official corpus is expected to
pass and the withdrawn one to fail; the script reports what it finds either way.

Writes ``artifacts/brain/grade_mapping_check.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from aura.common.config import ARTIFACTS

OFFICIAL = ARTIFACTS / "brain" / "embeddings_official" / "brats2020_369.json"
WITHDRAWN_CACHE = ARTIFACTS / "brain" / "cache"
OUTPUT = ARTIFACTS / "brain" / "grade_mapping_check.json"
N_PERMUTATIONS = 10_000
ALPHA = 0.01


def _auroc(y: np.ndarray, score: np.ndarray) -> float:
    pos = y.astype(bool)
    r = stats.rankdata(score)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def check(name: str, grade: list[str], enhancing: np.ndarray, tumour: np.ndarray,
          seed: int = 7) -> dict:
    lgg = np.asarray([g.lower() == "lgg" for g in grade])
    non_enh = enhancing == 0
    table = [[int((non_enh & lgg).sum()), int((~non_enh & lgg).sum())],
             [int((non_enh & ~lgg).sum()), int((~non_enh & ~lgg).sum())]]
    odds, p_fisher = stats.fisher_exact(table, alternative="greater")

    frac = enhancing / np.maximum(tumour, 1)
    # HGG is the positive class for "more enhancement".
    auc = _auroc(~lgg, frac)
    rng = np.random.default_rng(seed)
    null = np.asarray([_auroc(rng.permutation(~lgg), frac) for _ in range(N_PERMUTATIONS)])
    p_perm = float((np.sum(null >= auc) + 1) / (N_PERMUTATIONS + 1))

    passed = bool(p_fisher < ALPHA)
    return {
        "corpus": name,
        "subjects": int(len(grade)),
        "labelled_LGG": int(lgg.sum()),
        "labelled_HGG": int((~lgg).sum()),
        "non_enhancing_rate_LGG": round(float(non_enh[lgg].mean()), 4),
        "non_enhancing_rate_HGG": round(float(non_enh[~lgg].mean()), 4),
        "contingency_[[LGG_nonenh,LGG_enh],[HGG_nonenh,HGG_enh]]": table,
        "fisher_one_sided_p": float(p_fisher),
        "odds_ratio": None if not np.isfinite(odds) else round(float(odds), 3),
        "enhancing_fraction_auroc_for_HGG": round(auc, 4),
        "permutation_p": p_perm,
        "verdict": ("PASS: labelled grade tracks enhancement in the images"
                    if passed else
                    "FAIL: labelled grade is not associated with enhancement in the images"),
    }


def main() -> int:
    results = []

    side = json.loads(OFFICIAL.read_text(encoding="utf-8"))
    subs = side["subjects"]
    results.append(check(
        "BraTS 2020 official (datasets/brats2020, name_mapping.csv)",
        [s["grade"] for s in subs],
        np.asarray([s["voxels"]["enhancing"] for s in subs], dtype=float),
        np.asarray([sum(s["voxels"].values()) for s in subs], dtype=float)))

    manifest = json.loads((WITHDRAWN_CACHE / "manifest.json").read_text(encoding="utf-8"))
    grades, enh, tum = [], [], []
    for rec in manifest["subjects"]:
        study = json.loads((WITHDRAWN_CACHE / "studies" / f"{rec['subject_id']}.json")
                           .read_text(encoding="utf-8"))
        pcv = study["label"]["per_class_voxels"]
        grades.append(rec["grade"])
        enh.append(pcv["enhancing"])
        tum.append(sum(pcv.values()))
    results.append(check(
        f"withdrawn cohort ({manifest['corpus_root']})",
        grades, np.asarray(enh, dtype=float), np.asarray(tum, dtype=float)))

    out = {
        "purpose": ("Tests whether each corpus's grade labels are attached to the right "
                    "images, using contrast enhancement in the expert segmentation. "
                    "Test and threshold fixed before scoring; see module docstring."),
        "primary_test": "one-sided Fisher exact, non-enhancing rate LGG > HGG",
        "alpha": ALPHA,
        "results": results,
    }
    OUTPUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for r in results:
        print(f"{r['corpus']}\n  LGG non-enh {r['non_enhancing_rate_LGG']:.3f}  "
              f"HGG non-enh {r['non_enhancing_rate_HGG']:.3f}  "
              f"Fisher p={r['fisher_one_sided_p']:.3g}  "
              f"AUROC {r['enhancing_fraction_auroc_for_HGG']:.3f} "
              f"(perm p={r['permutation_p']:.3g})\n  {r['verdict']}")
    print(f"[done] {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
