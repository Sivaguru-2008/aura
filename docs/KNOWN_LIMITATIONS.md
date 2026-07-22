# AURA — Known Limitations & Exclusions

This document outlines the clinical, technical, and operational boundaries of the current AURA codebase.

---

## 1. Regulatory Status

* **Not a Validated Medical Device**: AURA is research-grade software. It has not received regulatory clearance or approval from the US Food and Drug Administration (FDA) as a Software as a Medical Device (SaMD), nor has it been certified under the EU Medical Device Regulation (MDR).
* **Clinical Decision Support Only**: AURA is designed strictly as a clinical decision support tool. It must never be used autonomously to establish diagnoses or direct patient treatments; a licensed clinician must always serve as the primary reader.

---

## 2. Technical & Model Limitations

### Closed-Vocabulary Diagnostics
* The evidence fusion and clinical reasoning engines operate on a closed vocabulary of **6 diagnoses**: Normal, COPD, Heart Failure, Malignancy, Pneumonia, and Pneumothorax.
* Visual findings outside the 7-label classifier head (e.g. fractures, chest tubes, mediastinal shifts, or specific vascular anomalies) are not represented and will be missed.

### OOD Energy-Score Boundaries
* The Out-of-Distribution (OOD) detector calculates an energy z-score based on the 8-dimensional evidence vector. Within valid chest radiograph space, this low-dimensional feature vector has limited dynamic range.
* Thus, the primary defense against non-radiograph images (e.g. hand X-rays, artifacts, or blank files) is the structural `xray_gate.validate_cxr()`, which runs at image intake. The energy-score OOD detector serves as a secondary defense.

### Conformal Prediction Saturation
* Although Mondrian conformal prediction ensures class-conditional coverage, classes with extremely low sample support (e.g. rare clinical cases) can cause the quantiles to saturate. 
* To prevent degeneracy, AURA implements a fallback to pooled marginal conformal thresholds, but this reduces class-specific precision for rare conditions.

---

## 3. Operational & Security Limitations

* **No Multi-User Isolation**: The FastAPI gateway is designed as a single-clinician offline service. It does not feature user session isolation, data partitioning, or multi-tenant database routing.
* **Security Seams**: By default, the service operates in an offline, unsecured mode trust-verifying the `x-aura-user` header. Although token authorization, rate-limiting, and MIME type-checks exist (`gateway/security.py`), they are turned off for the offline seeder demo.
