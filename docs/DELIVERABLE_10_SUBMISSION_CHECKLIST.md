# Scientific Deliverable 10: Journal Submission Readiness Checklist — npj Digital Medicine

**Manuscript Title:** Claimed quantum benefits in clinical machine learning could not be verified under symmetric evaluation: an empirical and literature audit  
**Target Journal:** *npj Digital Medicine* (Nature Portfolio)  
**Author:** Sivaguru R. M. (Independent Researcher)  
**Audit Date:** November 2026 (Revision R4)  
**Lead Auditor Role:** Scientific Manuscript Editor & Nature Portfolio Submission Specialist  

---

## 1. Compliance Matrix: Nature Portfolio & npj Digital Medicine Guidelines

| Requirement Item | Journal Guideline Specification | Manuscript Implementation Status | Verification Status |
|---|---|---|:---:|
| **Manuscript Format** | Original Research Article | Structured as an Original Research Article adhering to Nature Portfolio formatting standards. | **COMPLIANT** |
| **Title Length & Style** | Concise, descriptive, devoid of promotional jargon or acronyms. | *"Claimed quantum benefits in clinical machine learning could not be verified under symmetric evaluation: an empirical and literature audit"* (18 words). | **COMPLIANT** |
| **Abstract Length** | Unstructured, single paragraph, strictly $\le 200$ words. | Exactly **199 words**, completely unstructured, summarizing clinical motivation, symmetric protocol, empirical findings, multi-seed variance, class-balanced macro failure, and literature audit. | **COMPLIANT** |
| **Display Items** | Maximum of 8 display items (figures and tables combined) in main text. | Exactly **7 display items**: 4 Tables (Tables 1, 2, 3, 4) + 3 Figures (Figures 1, 2, 3). | **COMPLIANT** |
| **Structural Ordering** | Introduction $\to$ Results $\to$ Discussion $\to$ Methods $\to$ Declarations $\to$ References. | Exactly implemented. Methods appears *after* Discussion in both LaTeX and Markdown source files. | **COMPLIANT** |
| **Methods Section** | Detailed, reproducible, broken into logical subsections with explicit algorithms. | Detailed subsections covering Algorithm 1, Cohort Construction, Circuit Architecture, Simulation Verification, and Statistical Inference. | **COMPLIANT** |
| **Data Availability Statement** | Mandatory explicit statement detailing public/credentialed repository access and licenses. | Full statement with PhysioNet (MIMIC-CXR), Stanford ML Group (CheXpert), BraTS 2020, and UCI repositories specified with licenses. | **COMPLIANT** |
| **Code Availability Statement** | Mandatory statement with open repository URL, software license, and reproduction steps. | Released under Apache 2.0 license at `https://github.com/Sivaguru-2008/aura` with single-command reproduction (`python reproduce_paper.py`). | **COMPLIANT** |
| **Author Contributions** | Mandatory CRediT statement detailing roles. | Included: *"S.R.M. conceived the study, formulated the symmetric evaluation protocol, developed software, conducted experiments, and wrote the manuscript."* | **COMPLIANT** |
| **Competing Interests** | Mandatory explicit declaration of financial and non-financial interests. | Included: *"The author declares no competing financial or non-financial interests."* | **COMPLIANT** |
| **Main References** | Complete bibliographic metadata, numbered consecutively in citation order. | **35 verified references** formatted in Nature style with complete author lists, journal titles, volumes, pagination, and years. | **COMPLIANT** |
| **Supplementary Information** | Self-contained companion document with Supplementary Notes, Tables, and Figures. | 17-page PDF (`supplementary_information.pdf`) with 9 Supplementary Notes, 5 Supplementary Tables (including Table S5 clinical confusion matrices), 3 Supplementary Figures, and 18 Supplementary References. | **COMPLIANT** |
| **Reporting Standards Alignment** | Alignment with international prediction and trial reporting standards. | Aligned with TRIPOD+AI (clinical prediction reporting) and CONSORT-AI (interventions involving artificial intelligence). | **COMPLIANT** |

---

## 2. Display Item Verification Table

| Item Identifier | Format | Width | Title / Content | Verification Status |
|---|---|---|---|:---:|
| **Table 1** | Table | Two-column | Methodological audit of eligible published quantum machine learning studies in digital health. | **VERIFIED** |
| **Table 2** | Table | Two-column | Pre-specified factorial analysis isolating the evaluation mechanisms of apparent quantum advantage. | **VERIFIED** |
| **Figure 1** | Figure | Two-column | Forest plot of evaluation mechanisms and re-tested clinical claims. | **VERIFIED** |
| **Table 3** | Table | Two-column | Pre-specified confirmatory evaluation across 56,534 untouched clinical patients under default and tuned protocols (Panels A, B, C). | **VERIFIED** |
| **Figure 3** | Figure | Two-column | Pre-specified confirmatory evaluation across 56,534 held-out patients scored without selection. | **VERIFIED** |
| **Table 4** | Table | Two-column | Class-conditional log-likelihood decomposition across 56,534 held-out patients. | **VERIFIED** |
| **Figure 2** | Figure | Two-column | Mechanistic dissection of observed circuit performance and kernel geometry. | **VERIFIED** |

*Total Count:* 4 Tables + 3 Figures = **7 Display Items** (strictly $\le 8$).

---

## 3. Submission Package Inventory

The following files constitute the verified submission package:
1. `manuscript.pdf`: Primary publication-ready compiled PDF (18 pages, clean two-column format).
2. `manuscript.tex`: Authoritative LaTeX source code.
3. `MANUSCRIPT_FINAL_npjDM.md`: Synchronized Markdown mirror of the main manuscript.
4. `supplementary_information.pdf`: Companion Supplementary Information PDF (17 pages, clean table of contents, bookmarks).
5. `supplementary_information.tex`: Supplementary Information LaTeX source code.
6. `reproduce_paper.py`: Automated reproduction suite passing all 5 verification levels.
7. `audit/`: Complete audit trail containing Deliverables 5 through 11.

---

## 4. Final Editorial Sign-Off

The manuscript package is 100% complete, fully verified against on-disk evidence, formatted strictly to *npj Digital Medicine* requirements, and ready for formal submission.
