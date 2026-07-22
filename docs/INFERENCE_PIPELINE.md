# AURA — 9-Stage Inference Pipeline

This document provides a step-by-step description of the data flow and calculations that occur when a chest radiograph is analyzed by the AURA orchestrator.

---

## The 9-Stage Orchestration Flow

```
   [ Upload Pixels ]
           │
           ▼
1. Image Intake & Normalization
           │  (224x224 grayscale)
           ▼
2. Vision Engine (DenseNet-121)
           │  (7 sigmoids + 1024-d embedding)
           ▼
3. Evidence Encoding (Priors + Findings)
           │  (x in [0,1]^8)
           ▼
4. Evidence Fusion (VQC / PoE + Conflict Guard)
           │  (resolved_logits)
           ▼
5. Clinical Reasoning (Guideline LRs)
           │  (final_posterior)
           ▼
6. Safety & Conformal predictions (ACI + OOD)
           │  (SafetyAssessment)
           ▼
7. Saliency & Attribution Explainability
           │
           ▼
8. Expected Information Gain Recommender
           │
           ▼
9. Grounded Report & Memory Indexing
```

---

## Detailed Pipeline Stages

### Stage 1: Intake & Normalization
The gateway receives a file upload (JPEG, PNG, or DICOM) at `/v1/studies/upload`. 
* DICOM inputs undergo windowing (VOI-LUT) and monochrome inversion via `pydicom`.
* The image is resized to $224 \times 224$ using OpenCV's `INTER_AREA` interpolation.
* Pixel intensities are normalized using ImageNet statistics (mean 0.449, std 0.226).

### Stage 2: Vision Engine
The normalized image is passed to `VisionEngine.analyze()`. It runs a forward pass on the fine-tuned DenseNet-121 model:
* Evaluates 7 sigmoid logits for opacity, consolidation, pleural effusion, cardiomegaly, nodule, pneumothorax, and hyperinflation.
* Extracts the 1024-dimensional feature vector from the final average pooling layer as the *evidence embedding*.

### Stage 3: Evidence Encoding
The 7 vision probabilities and patient structured priors (age, smoking status) are passed to `services/fusion/evidence.py`. They are mapped to an 8-channel vector $x \in [0, 1]^8$, where channel indices are:
* `[0]` opacity, `[1]` consolidation, `[2]` pleural_effusion, `[3]` cardiomegaly, `[4]` nodule, `[5]` pneumothorax, `[6]` hyperinflation, `[7]` prior_risk composite.

### Stage 4: Evidence Fusion & Conflict Guard
`FusionEngine.fuse_vector()` evaluates the evidence vector $x$:
1. Computes the posterior differential over 6 diagnoses using the active backend (VQC by default).
2. Runs the classical Product-of-Experts (PoE) baseline in parallel.
3. **Wasserstein Conflict Guard**: Computes the Earth Mover's Distance between VQC and PoE. If the distance exceeds the threshold ($\tau = 0.12$), the guard triggers a fallback to classical PoE and flags high epistemic risk.
4. Outputs the `resolved_logits` representing the chosen backend.

### Stage 5: Clinical Reasoning
If multimodal laboratory values or patient symptoms (e.g. BNP, fever) are present, `ClinicalReasoner` nudges the posterior using guideline-specific log-likelihood ratios:
* Heart failure is revised upward if BNP > 400.
* Pneumonia is revised upward if fever is present.
Outputs the `final_posterior`. If no multimodal context is present, this equals the imaging-only posterior.

### Stage 6: Safety & Conformal Predictions
`SafetyEngine.assess()` validates the `final_posterior` and `resolved_logits`:
1. Calibrates raw logits using temperature scaling.
2. Computes the conformal prediction set using Mondrian quantiles and ACI online feedback.
3. Computes the energy score OOD z-score on the default backend's logits.
4. Determines if the system must **abstain** (low confidence, large conformal set, high epistemic uncertainty, or OOD).

### Stage 7: Explainability
`ExplainEngine.explain()` runs explainability routines on the study:
* **Grad-CAM++**: Generates heatmaps over the raw image to localize the region driving the top diagnosis.
* **Feature Attribution**: Calculates leave-one-out impact for each of the 8 evidence channels.
* **Counterfactuals**: Generates alternative scenarios ("if opacity was 0.0, pneumothorax probability increases").

### Stage 8: Recommendation Engine
`RecommendEngine.recommend()` ranks diagnostic next-steps by evaluating Expected Information Gain (EIG) over a 5-test catalog (e.g. Chest CT, Echocardiogram, Sputum culture):
* Measures expected entropy reduction in the fusion posterior if the test result was known.
* Edges are masked based on a clinical dependency graph to prevent recommending redundant tests.

### Stage 9: Grounded Report & Memory
`ReportEngine.compose()` writes a text report:
* Drafts findings, impression, and recommendations where every sentence is tied to supporting evidence.
* If the SafetyEngine abstained, the impression is locked to `INDETERMINATE — human review required`.
* The MemoryEngine indexes the 1024-d image embedding using cosine similarity to search for similar cases.
* All outputs are saved to the SQLite database.
