# AURA Known Limitations and Clinical Boundaries

This document outlines the known clinical boundaries, algorithmic constraints, and dataset biases of the AURA Clinical Intelligence Copilot.

---

## 1. Algorithmic and Structural Constraints

### Brain MRI (NeuroMind)
- **Sequence Dependencies:** NeuroMind requires a complete study with all four standard sequences (**FLAIR, T1, T1CE, and T2**). If any sequence is missing, the system will raise an incomplete study flag and abstain from providing tumor segmentation overlays or diagnostic probabilities.
- **Slice Resolution Floor:** The MRI Foundation pipeline requires a minimum of **2 slices** per series. A single-slice upload (2D image representing a 3D volume) does not carry through-plane spatial geometry and cannot be canonicalized or resampled, resulting in a clean rejection (`unreadable_image`).
- **Head CT Ambiguity:** Pixel geometry features (gray level distribution and background air-to-subject ratio) cannot distinguish between axial brain MRIs and axial head CTs. While DICOM metadata will safely route a head CT as unsupported, a raw PNG/JPEG export of a head CT will trigger the head geometry signature and route to NeuroMind, flagged for review.

### Chest X-Ray (Thorax)
- **Modality Scope:** AURA Thorax is strictly validated for **frontal chest radiographs**. It actively rejects lateral projections, extremities, abdominal x-rays, and mammograms during modality routing.
- **Color Interference:** Plain-image signatures reject colored screenshots or files containing annotations/color bars, as medical radiographs are grayscale.

---

## 2. Dataset Biases & Demographic Gaps

### Chest Model (MIMIC-CXR Validation)
- The DenseNet-121 model was trained on the MIMIC-CXR dataset, which represents a patient population from a tertiary academic medical center in the United States.
- **Prevalence Gaps:** Rare pathologies such as pneumothorax (prevalence: 1.8% in validation split) and hyperinflation (prevalence: 1.6%) suffer from high positive prediction set sizes due to sparse training signals.

### Brain Model (BraTS2020 Validation)
- The ResU-Net model was trained on the BraTS2020 dataset, which consists of preoperative brain MRIs of patients diagnosed with High-Grade Gliomas (HGG) or Low-Grade Gliomas (LGG).
- **Clinical Phase Bias:** The model is not validated for postoperative monitoring, radiation necrosis differentiation, or non-glial tumors (such as meningiomas, acoustic neuromas, or primary central nervous system lymphomas).

---

## 3. Clinical Safety Boundaries

- **Not a Diagnostic Replacement:** AURA is a clinical intelligence copilot designed to assist radiologists and physicians. It is not approved for autonomous primary diagnostic interpretation.
- **Safety Abstentions:** Under the default `community_conservative` policy, AURA abstains from a clinical claim and advises manual radiological review when **any** of the following holds:
  - epistemic uncertainty (mutual information between deep-ensemble members) exceeds `0.15`;
  - the OOD energy-score z-score exceeds `2.5`;
  - the 90% conformal set contains more than `4` labels;
  - the top calibrated probability falls below `0.3` (a weak call across six classes).

  These are the values the server actually runs; `aura/tests/test_doc_numbers.py::test_published_safety_thresholds_match_the_served_policy` fails the build if this list drifts from `get_settings()`. Selecting the `academic_aggressive` profile (`AURA_SAFETY_POLICY=academic_aggressive`) loosens the first three to `0.25` / `3.5` / min-coverage `0.50`.

  > **Historical note.** Versions of this document before 2026-07-30 published `0.45`, `3.0` and set size `> 3`. Those were tuned against an overconfident *synthetic* fusion model (temperature 0.77); once fusion was honestly calibrated (temperature 0.94) they abstained on roughly **91% of real films**, and were recalibrated to the values above. See the abstention-operating-point comment in `aura/common/config.py`.

- **Conformal Set Scannability:** Conformal prediction sets guarantee 90% coverage but routinely include multiple competing diagnoses when the presentation is ambiguous — at 90% coverage a genuinely uncertain six-class model needs sets of 4–6, and the measured mean set size is ~3.3–3.5. AURA commits to a single answer on roughly a third of films and defers the rest. Users must review every item in the set, not only the top-ranked one.
