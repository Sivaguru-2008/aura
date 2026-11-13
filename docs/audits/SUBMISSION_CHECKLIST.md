# Submission Checklist for npj Digital Medicine

**Manuscript Title:** Symmetric evaluation of shallow quantum machine learning in healthcare  
**Target Journal:** *npj Digital Medicine* (Nature Portfolio)  
**Author:** Sivaguru R. M. (Independent Researcher, Puducherry, India)  
**Audit Date:** 2026-11-02  
**Final Submission Status:** COMPLIANT / READY FOR FREEZE  

---

## 1. Editorial & Formatting Compliance

| Requirement | npj Digital Medicine Standard | Manuscript Status | Compliance Notes |
| :--- | :--- | :--- | :--- |
| **Title** | Concise, informative, non-promotional (<15 words) | **Compliant** | 8 words: "Symmetric evaluation of shallow quantum machine learning in healthcare". No hype or unproven claims. |
| **Abstract Structure** | Max 250 words; structured (Background, Methods, Results, Conclusions) | **Compliant** | Exactly structured with clear headings; word count: 247 words. Concludes on minority-class degradation and computational costs. |
| **Main Text Word Count** | Typically 3,500–5,000 words excluding Methods, Tables, References | **Compliant** | Main narrative is within limit (~4,400 words); extensive mathematical details, derivations, and full reproduction protocols are placed in Supplementary Information. |
| **Display Items** | Max 6–8 figures and tables in main text | **Compliant** | Exactly 7 display items in main text: 3 Figures (Fig 1: PRISMA + Forest; Fig 2: Confirmatory & Equivalence; Fig 3: Mechanism M1–M3) and 4 Tables (Table 1: Benchmark summary; Table 2: 7 Symmetric rules; Table 3: Confirmatory C1 & C2 results; Table 4: Class-conditional likelihood breakdown). |
| **Citation Format** | Nature numbered style (superscript, order of appearance) | **Compliant** | Vancouver/Nature numerical style implemented with `cite.sty` and BibTeX. |

---

## 2. Reporting Guidelines & Clinical AI Standards

| Guideline | Standard | Status | Specific Location / Evidence |
| :--- | :--- | :--- | :--- |
| **TRIPOD+AI** | Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis + AI | **Compliant** | Fully mapped in Supplementary Note 1. Pre-specified target population, predictor definitions, calibration curves, missing data handling, and participant flow accounting. |
| **PRISMA 2020** | Systematic literature screening flow and documentation | **Compliant** | Documented in Results Section 1, Figure 1a, and Supplementary Note 4. Explicitly discloses PubMed-only scoping search (231 records, 129 health-related, 41 advantage claims, 8 eligible, 5 reconstructible). |
| **Single-Rater Disclosure** | Screening conducted by single author | **Compliant** | Disclosed transparently in Methods, Results, and Supplementary Note 4. No synthetic consensus, Cohen's kappa, or dual-rater screening claimed. |
| **MI-CLAIM / CONSORT-AI** | Clinical machine learning validation standards | **Compliant** | Strict separation of development cohort ($n=8{,}000$) from confirmatory cohort ($n=56{,}534$); zero patient overlap ($|S_{\text{dev}} \cap S_{\text{test}}| = 0$); deterministic patient deduplication. |

---

## 3. Methodological & Statistical Rigor

| Audit Item | Verification Status | Verification Detail |
| :--- | :--- | :--- |
| **Equivalence Testing (TOST)** | **VERIFIED** | Schuirmann Two One-Sided Tests implemented with proper degrees of freedom: $\text{df}=56{,}533$ for patient cohort ($p_{\text{TOST}} < 10^{-6}$), $\text{df}=9$ for 10-seed paired comparison ($p_{\text{TOST}} = 2.57 \times 10^{-6}$). Margin $\pm 0.010$ nats justified as operational benchmark. |
| **Class-Conditional Breakdown** | **VERIFIED** | Population cancellation explicitly unpacked in Table 4 and Abstract. VQC degrades on Malignancy ($+0.1771$ nats), Heart Failure ($+0.0753$ nats), and Pneumothorax ($+0.0549$ nats). Macro-mean difference $+0.03892$ nats fails equivalence ($p_{\text{TOST}} > 0.99$). |
| **Calibration ECE Forensic Audit** | **VERIFIED** | Canonical 10-seed VQC calibration recomputed on matched partition: $\text{ECE} = 0.0117 \pm 0.0037$ ($\text{SE} = 0.0012$, range $0.0065\text{--}0.0163$). Single winning checkpoint: $\text{ECE} = 0.0111$. Discrepancy with post-hoc slice ($0.0252 \pm 0.0038$) completely explained and documented in Table 3 caption and Supplementary Note 6.2. |
| **Validation Cohort Dual-Use** | **VERIFIED** | Minor optimistic selection bias on development validation loss disclosed transparently in Methods and Supplementary Note 7.7. Zero test-set leakage. |
| **Hardware vs. Simulation** | **VERIFIED** | Explicitly stated that all quantum evaluations represent exact statevector classical simulations (double-precision $L_\infty$ error $< 7.78 \times 10^{-16}$). No unverified claims of physical QPU deployment. |

---

## 4. Open Science, Data & Code Availability

| Requirement | Audit Finding | Artifact Reference |
| :--- | :--- | :--- |
| **Data Availability Statement** | Complete accession protocols and persistent links for CheXpert-v1.0-small (Stanford AIMI), MIMIC-CXR-JPG (PhysioNet), TCGA-LGG/GBM (NIH GDC), BraTS 2020 (TCIA), and UCI datasets. | Section "Data Availability" in main text and Supplementary Note 7. |
| **Code Availability Statement** | Public GitHub repository URL specified. Complete code tree covering preprocessing, training, calibration, TOST testing, and figure generation. | Section "Code Availability" and Note 7. |
| **Executable Reproduction** | Level A verification script (`python reproduce_paper.py`) executes in $\sim 10$ seconds, passing all 31 numerical assertions and verifying 8 artifact SHA-256 hashes. Pytest suite passes 26/26 tests. | `reproduce_paper.py`, `aura/tests/test_doc_numbers.py`. |
| **Level C Retraining Instructions** | Full step-by-step instructions for 36-configuration grid retraining across all models provided in Supplementary Note 7.8. | Supplementary Note 7.8. |

---

## 5. Declarations & Governance

| Item | Status | Text Verified |
| :--- | :--- | :--- |
| **Competing Interests** | Fully Declared | Author declares no competing financial or non-financial interests. |
| **Ethics & IRB** | Fully Declared | Institutional review exemptions and third-party data use agreements (Stanford AIMI, PhysioNet credentialed access) fully documented. |
| **Funding** | Fully Declared | Independent research; no dedicated external grant funding to declare. |
| **Author Contributions** | Fully Declared | Sivaguru R. M. conceived the study, developed the software, executed all experiments, conducted the literature audit, performed the statistical analyses, and wrote the manuscript. |

---

## 6. Final Freeze Verification Sign-Off

- [x] Untouched backup archived (`manuscript_backup_freeze_r4/`).
- [x] Canonical PDF files built and mirrored to root (`FINAL_MANUSCRIPT.pdf`, `SUPPLEMENTARY_INFORMATION.pdf`).
- [x] Source files frozen in `FINAL_MANUSCRIPT_SOURCE/`.
- [x] All 24 core claims verified against raw artifacts (`VERIFIED_CLAIMS.csv`).
- [x] Zero unresolved fatal discrepancies.
