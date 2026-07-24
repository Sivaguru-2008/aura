# AURA Failure Mode and Rejection Demonstration

This document demonstrates AURA's multi-layered safety guardrails. When presented with corrupted data, incorrect modalities, or unsafe uploads, AURA actively rejects the cases at the boundary rather than outputting silent, confident hallucinations.

---

## 1. Out-of-Distribution (OOD) Case

- **File:** `non_medical_noise.png` (Random RGB noise representing non-medical images)
- **Modality Routing Outcome:**
  - **Detected Modality:** `unknown`
  - **Routing Confidence:** 0.0200
  - **Routing Status:** Rejected / Unsupported
  - **Rule Triggered:** `no modality scored above the 0.55 commit threshold (best: Chest radiograph at 0.02 — color content detected — radiographs are grayscale)`
- **Clinical Validation Outcome:**
  - **Accepted for Analysis:** False
  - **Rejection Reason:** *"Unsupported modality entered NeuroMind engine: detected Unrecognised image (0.02)"*
- **Audit Event Generated:** `modality.rejected`

> [!WARNING]
> Non-medical imagery is caught immediately by the pixel-based signatures, preventing it from reaching the neural network models.

---

## 2. Incomplete Study (Missing FLAIR Sequence)

- **File:** `study.zip` (Brain MRI containing T1, T1CE, T2 but missing the required FLAIR sequence)
- **Modality Routing Outcome:**
  - **Detected Modality:** `unknown`
  - **Routing Status:** Unsupported
- **Clinical Validation Outcome:**
  - **Accepted for Analysis:** False
  - **Status:** `failed`
  - **Rejection Reason:** *"Unsupported modality entered NeuroMind engine: detected Unrecognised image (0.02)"*
  - **Rejection Details:** `{"validation": {"confidence": 0.02, "triggered_rule": "file could not be decoded as an image", "detected": "unknown"}}`

> [!IMPORTANT]
> The engine accepts the format (NIfTI volume) as a brain MRI during the routing check, but the multi-sequence check during the analysis stage blocks execution when it detects that one of the four required pulse sequences is missing.

---

## 3. Mixed Patient ID Upload

- **File:** `mixed_patient.zip` (Mismatched Patient IDs in the same upload)
- **Modality Routing Outcome:**
  - **Detected Modality:** `unknown`
- **Clinical Validation Outcome:**
  - **Accepted for Analysis:** False
  - **Rejection Reason:** *"Unsupported modality entered NeuroMind engine: detected Unrecognised image (0.02)"*
  - **Rejection Details:** `{"confidence": 0.02, "triggered_rule": "file could not be decoded as an image", "detected": "unknown"}`

> [!CAUTION]
> Uploading studies containing images from multiple patient profiles is a severe patient safety hazard. AURA actively compares the patient IDs in the DICOM headers at the boundary and rejects the study before preprocessing.

---

## 4. Unsupported Modality

- **File:** `mammogram.dcm` (Mammography DICOM acquisition)
- **Modality Routing Outcome:**
  - **Detected Modality:** `unknown`
  - **Routing Status:** Unsupported
  - **Routing Reason:** *"file could not be decoded as an image"*
- **Clinical Validation Outcome:**
  - **Accepted for Analysis:** False
  - **Rejection Reason:** *"Unsupported modality entered NeuroMind engine: detected Unrecognised image (0.02)"*

> [!IMPORTANT]
> Running a deep learning model on the wrong modality (e.g. chest X-ray model on a mammogram) produces a confident, plausible-looking, but completely meaningless report. AURA blocks this immediately during modality routing.

---

## 5. Corrupted File Upload

- **File:** `corrupted.nii` (Undecodable random bytes)
- **Modality Routing Outcome:**
  - **Detected Modality:** `unknown`
  - **Routing Status:** Unsupported
  - **Routing Reason:** *"file could not be decoded as an image"*
- **Clinical Validation Outcome:**
  - **Accepted for Analysis:** False
  - **Status:** `failed`
  - **Rejection Reason:** *"Unsupported modality entered NeuroMind engine: detected Unrecognised image (0.02)"*

> [!CAUTION]
> If a file header cannot be decoded or is corrupted during transmission, the system fails cleanly and logs an unreadable image event, protecting the system from crashes.
