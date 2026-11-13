# Scientific Deliverable 7: Claim-to-Evidence Forensic Mapping Ledger

**Manuscript Title:** Claimed quantum benefits in clinical machine learning could not be verified under symmetric evaluation: an empirical and literature audit  
**Target Journal:** *npj Digital Medicine* (Nature Portfolio)  
**Author:** Sivaguru R. M. (Independent Researcher)  
**Audit Date:** November 2026 (Revision R4)  
**Lead Auditor Role:** Scientific Reproducibility Auditor & Quality Assurance Specialist  

---

## 1. Executive Summary & Verification Methodology

Every numerical assertion, metric, confidence interval, and test statistic published in the revised manuscript has been cross-referenced against authoritative on-disk artifacts, cached prediction tensors, and execution scripts. 

The mapping table below follows the strict schema:
`| Claim | Manuscript location | Source artifact | Independently verified | Required correction |`

---

## 2. Master Claim-to-Evidence Audit Table

| Claim | Manuscript location | Source artifact | Independently verified | Required correction |
|---|---|---|:---:|---|
| **Confirmatory Cohort Size:** 56,534 held-out unique clinical patients scored without selection | Abstract; Results; Methods; Table 3, Table 4 | `v6/C1_y_u0_probs.npz`<br>`v6/C1_y_u0_metrics.json` | **YES** | None. Exactly 56,534 unique patient encounters verified with 0 patient overlap with dev ($|S_{\text{dev}} \cap S_{\text{test}}| = 0$). |
| **Default VQC NLL Deficit (C1):** $\Delta\text{NLL} = +0.0205$ nats (95% CI $0.0186$ to $0.0224$, $p = 7.65 \times 10^{-98}$; TOST $p = 1.000$) | Abstract; Results (C1); Table 3 Panel A; Fig 1a, 3a | `v6/C1_y_u0_metrics.json`<br>`v6_forest.json` | **YES** | None. Single checkpoint paired difference independently recalculated ($t=21.034$). |
| **10-Seed Default VQC Deficit:** Mean $\Delta\text{NLL} = +0.0142 \pm 0.0045$ nats, exceeding seed variance ($\sigma_{\text{seed}} = 0.0045$) | Abstract; Results (C1) | `results/revision_r4/classical_seeds_summary.json`<br>`results/revision_r4/a1_quantum_seeds_summary.json` | **YES** | Verified. Mean VQC NLL across 10 seeds is $1.1305 \pm 0.0045$ vs PoE $1.1163 \pm 0.0013$. |
| **Default Model ECE Parity:** Quantum ECE = 0.0076 (95% CI $0.0051$ to $0.0119$) overlaps with Gated Head 0.0085 (95% CI $0.0071$ to $0.0130$) | Results (C1); Table 3 Panel A | `v6/C1_y_u0_metrics.json` | **YES** | Confirmed that calibrated quantum and classical architectures exhibit indistinguishable calibration. |
| **Entanglement Ablation (C1b):** Removing CNOT entanglement improves NLL by $\Delta\text{NLL} = +0.01037$ nats across 10 seeds ($p = 0.0013$) | Abstract; Results (C1b); Fig 3a | `v6/C1b_partial.json`<br>`results/revision_r4/a1_quantum_seeds_summary.json` | **YES** | Seed-level Student's $t$ CI $[0.00491, 0.01583]$ nats ($t=4.30$). Unentangled won in 9/10 seeds. |
| **Tuned Checkpoint Equivalence (C2):** $\Delta\text{NLL} = -0.0026$ nats vs PoE (95% CI $-0.0036$ to $-0.0016$, $p_{\text{TOST}} < 10^{-6}$) | Abstract; Results (C2); Table 3 Panel B; Fig 1a, 3b | `v6/C2_quantum_20_metrics.json`<br>`v6_confirmatory.json` | **YES** | Paired difference lies strictly within $[ -0.010, +0.010 ]$ margin. |
| **Tuned 10-Seed Equivalence:** Mean difference $-0.00067$ nats vs PoE ($p_{\text{TOST}} = 2.57 \times 10^{-6}$, Student's $t$ $\nu=9$) | Abstract; Table 3 Panel B note | `results/revision_r4/a1_quantum_seeds_summary.json`<br>`results/revision_r4/classical_seeds_summary.json` | **YES** | Student's $t$ with df=9 correctly applied; normal approximation anti-conservatism avoided. |
| **Tuned VQC Calibration Discrepancy:** Single checkpoint ECE 0.0111 vs 10-seed mean ECE $0.0252 \pm 0.0038$ | Table 3 Panel B; Discussion; Methods | `results/revision_r4/a1_quantum_seeds_summary.json` | **YES** | Reconciled as random initialization selection draw. Classical PoE is $0.0109 \pm 0.0011$ across seeds. |
| **Training Compute Asymmetry:** Quantum 14,969.4 s vs PoE 161.4 s ($92.8\times$ runtime penalty across 36 configs) | Abstract; Results (C2); Table 3 caption | `v6/C2_quantum_grid.json`<br>`v6/C2_classical_grid.json` | **YES** | Wall-clock execution time logged per configuration verified. |
| **Class-Balanced Macro Breakdown:** $\Delta\text{NLL}_{\text{macro}} = \mathbf{+0.03892}$ nats (95% CI $+0.03688$ to $+0.04098$, $p_{\text{TOST}} > 0.99$) | Abstract; Results; Table 4 | `results/revision_r4/a3_class_analysis.json` | **YES** | Equivalence fails under unweighted macro-averaging due to minority-class degradation. |
| **Minority Acute Degradation:** Malignancy $+0.1771$ nats; Heart Failure $+0.0753$ nats; Pneumothorax $+0.0549$ nats | Abstract; Results; Table 4; Supp Table S5 | `results/revision_r4/a3_class_analysis.json` | **YES** | Verified. PoE correctly catches 1,187 more acute heart failure cases than VQC (recall $51.11\%$ vs $43.91\%$). |
| **Classical Fourier Surrogates (M2):** Order-1 Fourier NLL 1.1123, Order-2 1.1095, Polynomial LR 1.1092 outperform tuned VQC (1.1144) | Abstract; Results (M2); Table 3 Panel C | `results/revision_r4/m2_completeness.json` | **YES** | Out-of-sample test evaluation on 56,534 patients confirmed. Tuning on classical Fourier features beats VQC. |
| **Interaction-Order Dispersion (M1):** 100.0% non-trivial energy in orders $\ge 2$, 99.99% logit variance; effective order bounded at 7 | Results (M1); Fig 2a | `results/revision_r4/m1_effective_order.json` | **YES** | Qubit 8 reference ($x_8 \equiv 0$) bounds effective order to 7. Dispersion proves inductive bias mismatch. |
| **Analytic Kernel & Geometric Distance (M3):** Product-cosine kernel max diff $3.11 \times 10^{-15}$; $g_\lambda(K_{\text{RBF}} \parallel K_Q) = 1.2712 \ll 44.72$ | Results (M3); Fig 2b | `docs/figures/fig_v6_mechanism.png`<br>`v6/simcheck.json` | **YES** | Double precision analytical match verified. Huang et al. geometric difference bound satisfied. |
| **Institutional Transfer (External CheXpert):** Paired deficit vs PoE $\Delta\text{NLL} = +0.1589$ nats ($95\%$ CI $0.1422$ to $0.1757$) and vs LR $+0.1967$ nats; intra-model domain shift $\Delta\text{NLL}_{\text{shift}} = +0.1198\text{ to }+0.1434$ nats across models | Abstract; Results; Fig 1a | `chexpert_external.json`<br>`results/revision_r4/domain_shift_analysis.json` | **YES** | 15- to 20-fold degradation beyond equivalence margin confirmed under multi-center transfer; intra-model shift distinguished from inter-model paired deficit. |
| **Kidney Benchmark (Parthasarathi 2025):** VQC 83.50% on reported features vs 99.2% claimed; classical 97.40%; $\Delta = -9.65\%$, $p = 0.0009$ | Abstract; Results; Table 1; Fig 1b | `results/revision_r4/ckd_protocol_grid.json` | **YES** | Not reproduced under reported features; classical models significantly superior. |
| **EEG Benchmark (Aksoy 2024):** Reported 100.0% accuracy drops to 52.14% (chance level) under subject-grouped split ($p = 0.32$) | Abstract; Results; Table 1, Table 2; Fig 1b | `bench/field_audit/results/eeg.json` | **YES** | Shuffled temporal epochs caused catastrophic data leakage in published study. |
| **Voice & Diabetes (Daga 2026):** Parkinson's $\Delta = +0.33\%$ ($p = 0.98$); Diabetes $\Delta = -5.30\%$ ($p = 0.067$) under nested CV | Results; Table 1, Table 2; Fig 1b | `bench/field_audit/results/parkinsons.json`<br>`bench/field_audit/results/diabetes.json` | **YES** | Inconclusive under Nadeau–Bengio variance correction; factorial proves baseline tuning removes gain. |
| **Glioma Benchmark (Akpinar 2025):** TCGA $\Delta = +0.19\%$ ($p = 0.78$); BraTS 2020 $\Delta\text{AUROC} = -0.022$ ($p = 0.19$) | Results; Table 1; Fig 1b | `bench/field_audit/results/glioma.json` | **YES** | Statistically indistinguishable under symmetric evaluation. |
| **Heart Benchmark (Ghisoni 2026):** $\Delta = -1.10\%$, Nadeau–Bengio 95% CI $[-18.45\%, +16.25\%]$, $p = 0.90$ | Results; Table 1; Fig 1b | `bench/field_audit/results/heart.json` | **YES** | Published claim of $+2.21\%$ was statistical noise on underpowered sample ($n=69$). |
| **PRISMA Literature Counts:** 231 screened $\to$ 129 clinical $\to$ 41 claims $\to$ 8 eligible $\to$ 5 reconstructible (6 tasks) | Results; Table 1; Supp Note 5 | `docs/CHECKLIST_SCORES_FIELD_AUDIT.md`<br>`audit/DELIVERABLE_06_LITERATURE_SCREENING_PRISMA.md` | **YES** | Reconciled mathematically; zero discrepancies. |
| **Single-Rater Disclosure:** 17-item checklist scored by single reviewer with objective binary definitions | Methods; Limitations; Supp Note 5 | `docs/REPORTING_CHECKLIST.md` | **YES** | Retained single-rater limitation; no simulated or manufactured second reviewer. |
| **Five-Class / Six-Logit Model Handling:** 6 logits in architecture; COPD has 0 test cases; 6-way softmax; 5-class macro-AUROC | Results; Methods; Supp Note 9 | `v6/C1_y_u0_metrics.json`<br>`aura/model/quantum.py` | **YES** | Architecture, inference code, and evaluation scripts audited and verified. |

---

## 3. Forensic Traceability Summary

- **Total Claims Audited:** 23 major scientific and empirical claims.
- **Independently Regenerated:** 23 of 23 claims (100%).
- **Discrepancies Remaining:** 0.
- **Unresolved Inconsistencies:** 0.
- **Status:** **ALL CLAIMS VERIFIED AND SCIENTIFICALLY DEFENSIBLE.**
