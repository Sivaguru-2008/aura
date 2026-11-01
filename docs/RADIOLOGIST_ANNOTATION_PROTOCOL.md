# Radiologist consensus annotation protocol

**Document:** `AURA-CLIN-PROT-RAD-CONSENSUS-v1` (draft)
**Goal:** replace the labels AURA derives from radiology reports with radiologist consensus
labels. Start with a held-out test set, then extend to training data. Measure what the
change does to accuracy and precision.

---

## 1. Why

AURA's chest labels come from a rule-based parser run over the free-text radiology report.
The parser agrees with 66 reference reads at κ 0.86 and F1 0.89. Those reference reads were
made by an AI, not by radiologists (`MARKET_COMPARISON.md` §1.3). So there is still no
radiologist-read ground truth. Labels derived from reports have two known weaknesses:

1. **Omissions.** Reports leave out incidental or stable findings, and those become false
   negatives in the training signal.
2. **Hedging.** A phrase like "cannot rule out minimal retrocardiac opacity" still has to
   become a yes or a no.

Macro precision at the served operating point is 0.26 (`MARKET_COMPARISON.md` §1.4). Part of
that comes from label noise. Part comes from deliberately sensitivity-first thresholds at
prevalences of 2–18%, and better labels will not change that part. How much precision
improves is something to measure, not something to promise.

A radiologist-labelled held-out test set is the precondition for any honest accuracy claim
and for any clearance attempt. The multi-reader multi-case (MRMC) study, which compares
readers with and without AURA, is a separate study that comes later. This protocol only
produces labels.

---

## 2. Panel

| Role | Count | Minimum credentials | Responsibility |
| :--- | :--- | :--- | :--- |
| **Primary readers (R1, R2)** | 2 | Board-certified diagnostic radiologist (ABR, FRCR or equivalent), 3+ years post-training | Independent, blinded annotation |
| **Adjudicator (R3)** | 1 | Subspecialty-trained (thoracic or neuroradiology), 10+ years experience | Resolves every finding R1 and R2 disagree on |
| **Quality lead** | 1 | Clinical AI scientist or medical physicist | Tracks agreement between and within readers |

---

## 3. Workflow

```
                   [ De-identified DICOM study ]
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       [ Reader 1 (blinded) ]          [ Reader 2 (blinded) ]
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                          [ Compare reads ]
                                 │
                 Do R1 and R2 agree on every finding?
                 ├──► YES: commit to the label set
                 └──► NO : route the disagreement to R3
                                 │
                                 ▼
                    [ Adjudicator R3 (blinded) ]
                                 │
                                 ▼
                         [ Final label ]
```

1. **Blinding.** Readers do not see patient identity, the original report, clinical
   outcomes or any AI output.
2. **Agreement.** When R1 and R2 agree on a finding's presence and location, that label is
   committed.
3. **Adjudication.** Every finding where R1 and R2 disagree goes to R3. R3 reads the image
   without knowing which reader said what. R3's call is final.

---

## 4. Agreement targets (proposed)

* **Metrics:** Fleiss' κ and Gwet's AC1.
* **Targets:** κ ≥ 0.80 for gross acute findings (pneumothorax, large effusion,
  cardiomegaly), and κ ≥ 0.75 for subtle findings (solitary nodule, early consolidation).
  Report what is achieved. Published agreement on subtle chest findings is often lower.
* **Consistency checks:** about 5% of cases are repeated as hidden controls, to measure
  each reader's agreement with themselves.

---

## 5. What readers record

### 5.1 Chest radiograph
For each finding:
1. **Presence:** present, absent or indeterminate.
2. **Confidence:** 1 = equivocal, 2 = probable, 3 = definite.
3. **Extent:** a bounding box or contour, laterality, and severity (mild, moderate or
   severe).

### 5.2 Brain MRI
1. Review the co-registered FLAIR, T1, T1ce and T2 volumes.
2. Voxel labels: 1 = necrotic or non-enhancing core, 2 = oedema, 3 = enhancing tumour.
3. Category: high-grade glioma, low-grade glioma, meningioma, metastasis or other.

---

## 6. Test-set rules

- Split by patient, and by site where possible, before anyone labels anything.
- No training, threshold-selection or calibration step reads the test set. The current chest
  model has no such set: its 2,099 validation images also chose the training epoch.
- Freeze the test set and record its hash in the model card.

---

## 7. Tools

Use a DICOM-native annotation tool such as MONAI Label or Labelbox. It needs:
- 16-bit windowing (lung, mediastinum and bone)
- viewing controls like the hospital's PACS
- export to JSON or NIfTI that `aura/ml/` can load

---

## 8. Data-access constraints (must be settled before recruiting)

- **MIMIC-CXR.** Under the PhysioNet credentialed-access data use agreement, images and
  reports may not be shared with anyone who does not hold their own credential. Every reader
  must therefore complete the PhysioNet credentialing process, including human-subjects
  training, and sign the MIMIC-CXR DUA, before seeing a single study. Annotation must happen in
  an environment that the DUA allows. Do not send images by email or upload them to a
  third-party labelling service unless that service is covered.
- **CheXpert.** Readers must agree to the Stanford research use agreement. The confirmatory
  cohort of manuscript v6 is CheXpert, so a CheXpert reader study speaks directly to the
  confirmatory test. A MIMIC reader study speaks only to the internal results.
- Record each reader's credential or agreement date in the annotation log.

## 9. Two studies this protocol supports (manuscript v6)

**9.1 Second reader for the 66-report finding reference standard.** One credentialed reader
reads the same 66 MIMIC-CXR reports blind to the existing reads and to the parser. Report
reader-vs-reader κ per finding and reader-vs-parser κ. This removes the single-reader
limitation of the finding labels.

**9.2 Clinician labels for the diagnostic target.** Two readers independently assign each
sampled study to one of the six (on CheXpert, five) diagnostic categories from the image,
blind to the report, the rule and every model output. Disagreements go to an adjudicator.
Report:

- agreement of each reader with the rule-derived target (Cohen's κ, with a bootstrap CI);
- reader–reader κ, which is the ceiling against which rule agreement should be judged;
- the fusion comparison of v6 C1, re-scored against adjudicated clinician labels. This is
  descriptive and was not part of the pre-registration.

**Sample size.** With 300 studies and κ near 0.6, the 95% CI half-width is about ±0.07. This
is enough to separate "rule ≈ clinicians" from "rule far below reader–reader agreement". Sample
studies stratified by rule category, about 50 each, with a fixed seed recorded *before* any
reader starts. A second-stage pre-registration must be committed before the labels are
compared with anything.
