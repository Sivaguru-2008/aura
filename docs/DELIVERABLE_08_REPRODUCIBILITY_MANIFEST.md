# Scientific Deliverable 8: Comprehensive Reproducibility Manifest & Artifact Provenance

**Manuscript Title:** Claimed quantum benefits in clinical machine learning could not be verified under symmetric evaluation: an empirical and literature audit  
**Target Journal:** *npj Digital Medicine* (Nature Portfolio)  
**Author:** Sivaguru R. M. (Independent Researcher)  
**Audit Date:** November 2026 (Revision R4)  
**Lead Auditor Role:** Scientific Reproducibility Auditor  

---

## 1. System and Execution Environment Specifications

The empirical analyses, simulations, and statistical tests reported in this work were executed under a strictly controlled computational environment:

| Component | Specification |
|---|---|
| **Operating System** | Microsoft Windows 11 Professional (x86_64) |
| **Python Runtime** | Python 3.14 (Virtual Environment located at `aura-main/.venv/`) |
| **LaTeX Engine** | MiKTeX 25.12 (`pdflatex`, e-TeX v3.141592653-2.6-1.40.26) |
| **Git Commit Reference** | `42a465d` (Branch: `revision-r4`) |
| **Public Repository** | `https://github.com/Sivaguru-2008/aura` |
| **Core Libraries** | PyTorch 2.3.0, NumPy 1.26.4, SciPy 1.13.0, Scikit-learn 1.4.2, Matplotlib 3.8.4, Pytest 8.2.0 |

---

## 2. Level A Pinned Cryptographic Manifest (Revision R4)

All 8 primary scientific evidence artifacts generated for Revision R4 in `results/revision_r4/` are pinned with SHA-256 cryptographic hashes in `manifest.json`. Every artifact has been verified bit-for-bit against its hash:

| Artifact Filename | File Size | Cryptographic SHA-256 Checksum | Scientific Content |
|---|:---:|---|---|
| `classical_seeds_summary.json` | 7,349 bytes | `04fff45a11eb9cff28114be261c14041b344588258e7b15d66f4458021dcfcbe` | Multi-seed re-training distribution for classical PoE and Gated Head baselines across 10 random seeds on 56,534 patients. |
| `a1_quantum_seeds_summary.json` | 9,395 bytes | `07c7778c35d985a1163ffcb975f73c4d5bce13540e1f729e2fa8a10738a927cb` | Multi-seed re-training distribution for Variational Quantum Circuits (default C1, ablation C1b, and tuned C2) across 10 seeds. |
| `a2_selection_robustness.json` | 48,226 bytes | `301193f7a10a08e15147814a1a3b8398b96434444c10712217c24a6fa189c46d` | Full 60-configuration classical grid sensitivity proof, confirming classical baselines converged at $C=0.01$ (NLL 1.1170). |
| `a3_class_analysis.json` | 6,331 bytes | `ee0d9ea07ef0f6d634dbfa87c8008a9f6356763c3a9f0a996da6c8c4a16ce59b` | Class-conditional log-likelihood decomposition and macro-averaging breakdown across 56,534 held-out patients. |
| `domain_shift_analysis.json` | 24,637 bytes | `12e548be2377488880d603a1fc6c52a3fc263304193fe4cff618ee5c87c88085` | Multi-center domain transfer evaluation from MIMIC-CXR to CheXpert, quantifying angle encoding failure under feature scale drift. |
| `m1_effective_order.json` | 11,763 bytes | `efeb24b60d0370d0fafe7df9f2e30f14371bb1aa081e74f8812739b69b61d36c` | Fourier interaction-order energy spectrum decomposition across 10 seeds, proving 100.0% of non-trivial energy resides in orders $\ge 2$. |
| `m2_completeness.json` | 1,583 bytes | `5a6dd9958ef5be1703080fffc0fec21b66804da64c20b8ec5d1b714b7e8d251a` | Classical Fourier and polynomial basis surrogate evaluations on 56,534 patients, demonstrating classical surrogacy. |
| `ckd_protocol_grid.json` | 48,608 bytes | `57c4f1ea768297b4ecba8736d589f81d856d11e5dc422ffb62553b30fe94d76f` | Re-test grid for Chronic Kidney Disease benchmark (Parthasarathi et al. 2025), isolating global leakage from in-fold tuning. |

---

## 3. Input Evidence and Prediction Provenance

Input evidence files audited in `aura.ml.evaluation.provenance` (`evidence_ledger.json`):
1. `quantum_study_evidence.npz`:
   - Feature representations extracted from frontal radiographs via DenseNet-121 backbone.
   - Pinned SHA-256 verified.
2. `fusion_matched_optimiser_predictions.npz`:
   - Predictions from matched L-BFGS-B optimization experiments ($n=447$).
   - Pinned SHA-256 verified.
3. `v6/C1_y_u0_probs.npz`:
   - Full 6-way uncalibrated and calibrated prediction probabilities for all 56,534 held-out patients across 4 model backends.
   - SHA-256 verified.

---

## 4. Single-Command Reproduction Protocol

### Quick Verification Suite:
To execute input hash verification, figure rebuilding, numerical drift testing, and Revision R4 Level A manifest verification:
```powershell
cd e:\AURA\aura-main
.\.venv\Scripts\python.exe reproduce_paper.py
```
**Expected Execution Time:** $\approx 8$ seconds.  
**Success Output:** `All 5 steps passed in 8s.`

### Full Regeneration Suite:
To regenerate all cross-validation folds, hyperparameter searches, and bootstrap intervals from cached evidence:
```powershell
cd e:\AURA\aura-main
.\.venv\Scripts\python.exe reproduce_paper.py --full
```
*(Requires credentialed local paths for MIMIC-CXR and CheXpert images).*

---

## 5. Data Access and Credentialing Requirements

The datasets analyzed in this study are subject to ethical and credentialed access agreements:
1. **MIMIC-CXR-JPG (v2.0.0):**
   - Access controlled by PhysioNet under the Credentialed Health Data License 1.5.0.
   - Requires CITI Human Subjects Training ("Data or Specimens Only Research").
   - Access URL: `https://physionet.org/content/mimic-cxr-jpg/2.0.0/`
2. **CheXpert-v1.0-Small:**
   - Access governed by the Stanford University Research Data Agreement.
   - Non-commercial academic research registration required.
   - Access URL: `https://stanfordmlgroup.github.io/competitions/chexpert/`
3. **BraTS 2020 Glioma Benchmark:**
   - Managed by the Medical Image Computing and Computer Assisted Intervention (MICCAI) Society.
   - Open registration with user agreement: `http://braintumorsegmentation.org/`
4. **UCI Machine Learning Repository Datasets:**
   - Publicly available without credentials:
     * Chronic Kidney Disease: `https://archive.ics.uci.edu/dataset/336/chronic_kidney_disease`
     * Parkinson's Disease Voice Phonation: `https://archive.ics.uci.edu/dataset/174/parkinsons`
     * Pima Indian Diabetes: `https://archive.ics.uci.edu/dataset/34/diabetes`
     * Heart Disease: `https://archive.ics.uci.edu/dataset/45/heart+disease`
