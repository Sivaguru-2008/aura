"""Recalibrate OOD energy statistics on real MIMIC-CXR films.

The shipped ``safety.npz`` computed its OOD mean/std on the synthetic training
world, so every *real* film scores as out-of-distribution and the safety engine
abstains — the deferred limitation noted in ``mimic/seed.py``. This script lifts
it: it runs the vision → evidence → fusion path over real MIMIC films *and* a
fresh synthetic batch, then refits ``ood_mean``/``ood_std`` on the combined
energy pool so both worlds are in-distribution. Genuinely aberrant studies still
land in the tail and abstain; the low-confidence / conformal-set / epistemic
abstention gates are untouched.

Run:  python -m ml.training.recalibrate_ood [n_real] [n_synth]
The previous calibration is backed up beside the artifact before overwriting.
"""
from __future__ import annotations

import shutil
import sys

import numpy as np

from common.config import ARTIFACTS, get_settings
from common.mathx import energy_score
from ml.data import make_sample
from mimic.patient import iter_patients
from services.fusion import FusionEngine
from services.fusion.evidence import encode
from services.safety.calibration import Calibration
from services.vision import VisionEngine


def _energy(vision_engine, fusion, temperature, study) -> float:
    img = np.array(study.image, dtype=float).reshape(study.image_shape)
    v = vision_engine.analyze(study.study_id, img)
    x = encode(v, study.priors)
    return energy_score(fusion.model.logits(x), temperature)


def run(n_real: int = 150, n_synth: int = 200) -> Calibration:
    vision = VisionEngine.load()
    fusion = FusionEngine()
    cal = Calibration.load()

    real: list[float] = []
    for split in ("validate", "train"):
        if len(real) >= n_real:
            break
        for patient in iter_patients(split, limit=4000):
            if patient.n_studies == 0:
                continue
            try:
                study = patient.to_study_input(study_index=-1)
            except (ValueError, OSError):
                continue
            real.append(_energy(vision, fusion, cal.temperature, study))
            if len(real) >= n_real:
                break
    if len(real) < 30:
        raise SystemExit(f"only {len(real)} real films found — corpus missing?")

    rng = np.random.default_rng(11)
    synth: list[float] = []
    from ml.data import IMG
    from schemas.clinical import DIAGNOSES
    from schemas.contracts import StudyInput
    for i in range(n_synth):
        s = make_sample(DIAGNOSES[int(rng.integers(len(DIAGNOSES)))], rng)
        synth.append(_energy(vision, fusion, cal.temperature, StudyInput(
            study_id=f"recal-{i}", image=[float(v) for v in s.image.flatten()],
            image_shape=(IMG, IMG), priors=s.priors)))

    pool = np.array(real + synth, dtype=float)
    thr = get_settings().ood_energy_threshold
    old_z = lambda e: (np.array(e) - cal.ood_mean) / cal.ood_std  # noqa: E731

    backup = ARTIFACTS / "safety.synthetic-ood.bak.npz"
    if not backup.exists():
        shutil.copy(ARTIFACTS / "safety.npz", backup)

    print(f"real films: n={len(real)}  old-z median {np.median(old_z(real)):+.2f} "
          f"(abstain>{thr})   synthetic: n={len(synth)} "
          f"old-z median {np.median(old_z(synth)):+.2f}")
    cal.ood_mean = float(pool.mean())
    cal.ood_std = float(pool.std() + 1e-6)
    cal.save()
    new_z = lambda e: (np.array(e) - cal.ood_mean) / cal.ood_std  # noqa: E731
    print(f"new ood_mean={cal.ood_mean:.4f} ood_std={cal.ood_std:.4f}")
    print(f"real-film new-z: median {np.median(new_z(real)):+.2f}, "
          f"ood rate {(new_z(real) > thr).mean():.1%}")
    print(f"synthetic new-z: median {np.median(new_z(synth)):+.2f}, "
          f"ood rate {(new_z(synth) > thr).mean():.1%}")
    print(f"backup of previous calibration: {backup.name}")
    return cal


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:3]]
    run(*args)
