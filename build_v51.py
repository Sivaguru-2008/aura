import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

repo_root = Path("e:/AURA/aura-main")
v5_path = repo_root / "docs" / "MANUSCRIPT_v5.0.md"
text = v5_path.read_text(encoding="utf-8")

edits = []

# --- E1: Title ---
old_title = "# Quantum advantage claims in clinical machine learning do not survive a symmetric comparison: six negative results from a deployed decision-support system"
new_title = "# Auditing quantum machine learning in digital health: one deployed system and five published claims under symmetric evaluation"
assert old_title in text, "old_title not found"
text = text.replace(old_title, new_title, 1)
edits.append(("E1 Title", old_title, new_title, "User prompt E1"))

# --- E14 (part 1): Remove Erratum banner ---
banner_start = text.index("> **Resolution of 13 September 2026 Erratum")
banner_end = text.index("## Abstract")
old_erratum = text[banner_start:banner_end]
new_erratum = ""
text = text[:banner_start] + text[banner_end:]
edits.append(("E14 Erratum banner removal", old_erratum, "[Banner removed; relegated to defect register in Methods]", "User prompt E14"))

# --- E2: Abstract Results and Conclusions ---
abs_start = text.index("**Results.** Six claims were examined; none survived.")
abs_end = text.index("**Keywords:**", abs_start)
old_abs_text = text[abs_start:abs_end].strip()

new_abs_text = (
    "**Results.** We examined six claims of quantum benefit: five by experiment, one by source-level audit. "
    "None survived. A reported calibration advantage (expected calibration error 0.020 versus 0.276) was traced to "
    "temperature scaling applied only to the quantum arm. Under one optimiser protocol for all four fusion backends, "
    "no significant ordering between the quantum circuit and the product-of-experts remained "
    "(\u0394 NLL = 0.0033 in the product-of-experts' favour, Holm p = 1.0); the advantage vanishes. "
    "At matched capacity on seven informative channels, removing entanglement improved negative log-likelihood in "
    "ten of ten seeds (Holm p = 0.006; seed-level variation, study-level uncertainty in \u00a72.5). "
    "Across a 48-cell architectural sweep, the served design was not supported: the best-performing cell carries zero entanglement. "
    "For glioma grading, on the 111 subjects held out from encoder training, the quantum kernel's AUROC was 0.0217 below an "
    "RBF kernel (0.9407 versus 0.9625, 95% CI \u22120.0543 to +0.0109; DeLong p = 0.191); the interval does not exclude a difference "
    "in either direction. Frozen and transferred to a second institution under pre-registration, the circuit ranked last of four on "
    "every metric, and one of our own pre-registered hypotheses was falsified. Of five published quantum-advantage claims audited under "
    "nested classical selection, none was supported; one reversed significantly, and two were traced to preprocessing or splitting leakage. "
    "A shot-budget mechanism we had proposed was retracted after source audit.\n\n"
    "**Conclusions.** In this system and these five studies, no quantum component showed a benefit that survived a second baseline, "
    "matched optimisation, or a second institution. A positive control shows the same comparison can detect a nonlinear advantage, "
    "including one for the quantum circuit. These results concern one encoding family, 8\u201310 qubits, and exact simulation, and do not "
    "rule out advantage elsewhere.\n\n"
)
text = text.replace(old_abs_text, new_abs_text.strip(), 1)
edits.append(("E2 Abstract Results & Conclusions", old_abs_text, new_abs_text, "User prompt E2"))

# --- E3: Intro - Gupta sentence and >80% deletion ---
old_gupta = "Gupta and colleagues [1] screened 4,915 records published between 2015 and October 2024, identified 169 eligible studies, and synthesised 16. They report that measured differences between quantum and classical methods are negligible, and that the results show no consistent trend supporting empirical quantum utility in digital health."
new_gupta = "Gupta et al. [1] screened 4,915 records, identified 169 eligible studies, and synthesised 16, finding negligible quantum\u2013classical differences. They also report ad-hoc comparator selection and that 81.7% of eligible studies used ideal simulation only."
assert old_gupta in text, "old_gupta not found"
text = text.replace(old_gupta, new_gupta, 1)
edits.append(("E3 Intro Gupta sentence", old_gupta, new_gupta, "User prompt E3"))

old_intro_80 = "**over 80% of peer-reviewed digital health QML papers claim empirical quantum superiority precisely within the 4\u201312 qubit regime** [1]."
new_intro_80 = "[PENDING: count of qualifying studies in our 231-record PubMed screen using 4\u201312 qubits, from screening log]."
assert old_intro_80 in text, "old_intro_80 not found"
text = text.replace(old_intro_80, new_intro_80, 1)
edits.append(("E3 Intro >80% claim removal", old_intro_80, new_intro_80, "User prompt E3"))

# --- Reordering & Numbering for Results: Front-load Field Audit (E11) ---
sec29_start = text.index("### 2.9 Field-wide audit: published medical QML advantages fail under symmetric evaluation")
sec29_end = text.index("## 3. Discussion")
sec29_text = text[sec29_start:sec29_end].strip()

# Remove section 2.9 from its current place
text_without_sec29 = text[:sec29_start] + text[sec29_end:]

new_table1_audit = (
    "**Table 1.** Field-wide literature audit of published medical QML claims evaluated under symmetric benchmarking across 25 outer folds (`bench/field_audit/results/`).\n\n"
    "| Study & Cohort | Target & Ground Truth | Reported Claim | Failure Mechanism | Symmetric Quantum | Symmetric Classical (nested-selected: 5 families, 16 configs) | \u0394 (Quantum \u2212 Classical) | Nadeau\u2013Bengio *p* | Audit Verdict |\n"
    "|---|---|---|---|---:|---:|---:|---:|---|\n"
    "| **CKD** (*PeerJ* 2025) | Chronic kidney disease (biochemical assays) | VQC 99.2% vs SVM 92.8% | Suboptimal classical baseline | 87.70% | **97.35%** | \u22129.65% [\u221214.9, \u22124.4] | **0.000915** | **REVERSED** |\n"
    "| **Glioma** (*Sci Rep* 2025) | Histological grade (TCGA molecular pathology) | VQC 83.0% vs SVM 75.0% | Suboptimal classical baseline | 85.60% | 85.41% | +0.19% [\u22121.2, +1.6] | 0.7824 | **NOT SUPPORTED** |\n"
    "| **Parkinson's** (*Sci Rep* 2026) | Motor exam & acoustic biomarkers | Stack spec. 81.3% vs SVM 58.5% | Data leakage (preprocessing: PCA across splits) | 43.33% | 43.00% | +0.33% [\u221225.0, +25.6] | 0.9785 | **NOT SUPPORTED** |\n"
    "| **Diabetes** (*Sci Rep* 2026) | Oral glucose tolerance test (Pima) | Stack recall 62.1% vs SVM 55.3% | Suboptimal classical baseline (omitted L2 penalty) | 69.34% | **74.64%** | \u22125.30% [\u221211.0, +0.39] | 0.0666 | **NOT SUPPORTED** |\n"
    "| **Schizophrenia** (*J Med Syst* 2024) | Psychiatric DSM clinical EEG | QSVM 100% vs classical 100% | Data leakage (partitioning: repeated epochs across splits) | 59.10% | **66.71%** | \u22127.60% [\u221223.0, +7.8] | 0.3187 | **NOT SUPPORTED** |"
)

new_sec21_text = (
    "### 2.1 Field-wide audit: published medical QML advantages fail under symmetric evaluation\n\n"
    "To determine whether the null and negative findings in AURA represent an idiosyncratic property of our architecture or a general property of applied medical QML, we conducted an empirical literature audit following a pre-registered protocol (`docs/FIELD_AUDIT_PROTOCOL.md`). We systematically screened 231 PubMed records (2019\u20132026) evaluating quantum classifiers on clinical prediction tasks, identifying peer-reviewed studies that claimed quantum superiority with fully specified, reproducible pipelines on openly accessible biomedical cohorts.\n\n"
    "[PENDING: screening flow 231 -> eligible -> reproducible -> 5 with counts from the screening log]. "
    "Because these five studies were isolated strictly on the criterion of computational reproducibility from available artifacts, they represent a feasibility-selected audit of reproducible claims rather than an unbiased random sample of the entire digital-health QML literature.\n\n"
    "Each study was audited under two arms: (1) a **reproduction arm** implementing the study's own claimed pipeline, and (2) a **symmetric arm** evaluating the quantum model against a nested-selected classical comparator (5 families, 16 configurations: logistic regression, RBF-SVM, random forest, histogram gradient boosting, MLP) across 25 outer cross-validation folds under leak-free preprocessing and Nadeau\u2013Bengio corrected resampling tests.\n\n"
    "Across the audited cohorts, spanning chronic disease laboratory assays, clinical neurological exams, and neuro-oncology (Table 1), **zero of five published claims of quantum advantage survived symmetric evaluation**:\n\n"
    "1. **Chronic Kidney Disease (CKD)** (*PeerJ Comput Sci* 2025; PMID 39958742). The authors claimed that a variational quantum classifier (VQC) achieved 99.2% accuracy, significantly outperforming classical SVM (92.8%), Random Forest (90.7%), and MLP (94.8%) on serum creatinine, blood urea, and urinalysis biomarkers. Under our symmetric protocol, every classical baseline improved dramatically to 96.0%\u201397.4% under standard hyperparameter grids, while the VQC reached only 87.7% accuracy. The claimed quantum advantage **reversed with high statistical significance in favour of classical models** (\u0394 = \u22129.65%, 95% CI: \u221214.9% to \u22124.4%, t = \u22123.78, p = 0.000915).\n"
    "2. **Glioma Grading (TCGA)** (*Sci Rep* 2025; PMID 40676161). The study reported that a VQC outperformed classical algorithms (KNN, SVC, DTC, RFC) with a cross-validated accuracy of 0.83 on 20 molecular and clinical features. Under symmetric cross-validation and matched tuning, Quantum VQC achieved 85.6% accuracy against 85.4% for classical models (\u0394 = +0.19%, 95% CI: \u22121.2% to +1.6%, t = 0.28, p = 0.7824); the claim is **not supported**.\n"
    "3. **Parkinson's Disease Progression** (*Sci Rep* 2026; PMID 42286086). The study claimed an adaptive quantum-kernel stack increased diagnostic specificity from 58.5% (classical RBF-SVM) to 81.3% on clinical acoustic voice biomarkers. A source-level audit identified that PCA dimensionality reduction had been fitted across the entire dataset prior to train/test partitioning, leaking label-correlated variance. When PCA was strictly encapsulated within training folds under our symmetric protocol, quantum and classical specificities were identical (43.3% versus 43.0%, \u0394 = +0.33%, 95% CI: \u221225.0% to +25.6%, t = 0.027, p = 0.9785); the claim is **not supported**.\n"
    "4. **Type-2 Diabetes (Pima)** (*Sci Rep* 2026; PMID 42286086). The same quantum-kernel stacking framework claimed to increase recall from 55.3% to 62.1% on glucose tolerance and metabolic criteria. Under leak-free symmetric evaluation, classical models attained 74.6% sensitivity while the quantum stack attained 69.3% (\u0394 = \u22125.30%, 95% CI: \u221211.0% to +0.39%, t = \u22121.92, p = 0.0666); the claim is **not supported**.\n"
    "5. **Adolescent Schizophrenia EEG** (*J Med Syst* 2024; PMID 38441727). The authors reported 100.0% classification accuracy using a Pauli-feature-map QSVM across EEG sensor channels. Source audit revealed that 100% accuracy arose from random slice-level splitting across repeated EEG epochs from the same subjects. When re-evaluated under subject-disjoint grouped cross-validation, performance collapsed toward chance for both modalities (59.1% quantum vs 66.7% classical, \u0394 = \u22127.60%, 95% CI: \u221223.0% to +7.8%, t = \u22121.02, p = 0.3187); the claim is **not supported**.\n\n"
    f"{new_table1_audit}\n\n---\n\n"
)

results_pos = text_without_sec29.index("## 2. Results\n\n") + len("## 2. Results\n\n")
text = text_without_sec29[:results_pos] + new_sec21_text + text_without_sec29[results_pos:]
edits.append(("E11 Front-loaded field audit to §2.1", "Old §2.9 at end of Results", "New §2.1 at start of Results with leakage separation column and nested-selected label", "User prompt E11"))

# Section renumbering
text = text.replace("### 2.1 A calibration advantage", "### 2.2 A calibration advantage", 1)
text = text.replace("### 2.2 The verdict depends on the comparator", "### 2.3 The verdict depends on the comparator", 1)
text = text.replace("### 2.3 The comparison detects a nonlinear advantage", "### 2.4 The comparison detects a nonlinear advantage", 1)
text = text.replace("### 2.4 Entanglement costs likelihood at matched capacity", "### 2.5 Entanglement costs likelihood at matched capacity", 1)
text = text.replace("### 2.5 The design optimum lies at zero entanglement", "### 2.6 Sensitivity across 48 architectural configurations", 1)
text = text.replace("### 2.6 The quantum kernel does not differ from a classical kernel", "### 2.7 The quantum kernel does not differ from a classical kernel", 1)
text = text.replace("### 2.7 External validation falsifies one of our own hypotheses", "### 2.8 External validation falsifies one of our own hypotheses", 1)
text = text.replace("### 2.8 A capability claim retracted on source-level audit", "### 2.9 A capability claim retracted on source-level audit", 1)

# Correct Table 1 (published block) Gated head macro AUROC mismatch from 0.8152 to 0.8056!
old_table1_gated_row = "| | Gated head | 126 | 0.887 | 0.5526 | 1.2663 | 0.0902 | 0.8152 |"
new_table1_gated_row = "| | Gated head | 126 | 0.887 | 0.5526 | 1.2663 | 0.0902 | 0.8056 |"
assert old_table1_gated_row in text, "old_table1_gated_row not found"
text = text.replace(old_table1_gated_row, new_table1_gated_row, 1)
edits.append(("Table 1 (now Table 2) Gated head published Macro AUROC correction", "0.8152", "0.8056", "fusion_baselines_n447.json, backends.learnable.macro_auroc"))

# Renumber Tables:
text = text.replace("**Table 1.** Fusion backend comparison", "**Table 2.** Fusion backend comparison", 1)
text = text.replace("(Table 1, upper block)", "(Table 2, upper block)", 1)
text = text.replace("Table 1's lower block", "Table 2's lower block", 1)
text = text.replace("(Table 1)", "(Table 2)", 1)

text = text.replace("(Table 2, Figure 4)", "(Table 3, Figure 4)", 1)
text = text.replace("**Table 2.** External validation on CheXpert", "**Table 3.** External validation on CheXpert", 1)

text = text.replace("**Table 3.** The six claims, and what each survived.", "**Table 4.** The six claims, and what each survived.", 1)
text = text.replace("row 5 of Table 3", "row 5 of Table 4", 1)

text = text.replace("Table 4 gives the result.", "Table 5 gives the result.", 1)
text = text.replace("**Table 4.** Positive control.", "**Table 5.** Positive control.", 1)

# --- E4: §2.3 (old §2.2) Reversal text & TOST ---
old_rev_sentence = (
    "The quantum circuit's published advantage over the product-of-experts does not merely lose significance; "
    "it **reverses in sign**, the product-of-experts now attaining the lower negative log-likelihood "
    "(Δ = 0.0033 in its favour, Holm *p* = 1.0)."
)
new_rev_sentence = (
    "The quantum circuit's published advantage over the product-of-experts disappears "
    "(Δ = 0.0033 in the product-of-experts' favour, Holm p = 1.0). "
    "We do not interpret this as a reversal; the estimate is compatible with zero. "
    "To assess equivalence, we ran a two one-sided test with a margin of ±[PENDING: margin to be pre-registered in an addendum before execution, e.g. 0.01 NLL]: "
    "[PENDING result]. Until that test is reported, we say only that no difference was detected."
)
assert old_rev_sentence in text, "old_rev_sentence not found"
text = text.replace(old_rev_sentence, new_rev_sentence, 1)
edits.append(("E4 §2.3 Reversal wording and TOST", old_rev_sentence, new_rev_sentence, "User prompt E4"))

# Figure 1 caption reversal wording:
old_fig1_rev = "The quantum circuit's advantage over the product-of-experts reverses."
new_fig1_rev = "The quantum circuit's likelihood advantage over the product-of-experts disappears (Δ NLL = 0.0033 in PoE's favour, Holm p = 1.0; no difference detected)."
assert old_fig1_rev in text, "old_fig1_rev not found"
text = text.replace(old_fig1_rev, new_fig1_rev, 1)

# Discussion paragraph 2 reversal wording:
old_disc_rev = "under one optimiser protocol, the published quantum advantage reverses in sign (§2.2)."
new_disc_rev = "under one optimiser protocol, the published quantum advantage vanishes (Δ = 0.0033, Holm p = 1.0; §2.3)."
assert old_disc_rev in text, "old_disc_rev not found"
text = text.replace(old_disc_rev, new_disc_rev, 1)

# --- E5: §2.4 (old §2.3) T1 paragraph and pairwise deltas ---
old_t1_para = (
    "On the linear target T1, the two linear arms lead and the quantum circuit is significantly worst on both accuracy "
    "(Δ = −0.036 against the logistic regression, Holm *p* = 0.0007) and likelihood (Δ = +0.092, Holm *p* < 10⁻⁴) — "
    "the outcome §2.2 predicts. On both interaction targets, the gated head — an ordinary classical nonlinearity — beats the "
    "best linear arm on negative log-likelihood by **0.057** (T2a) and **0.060** (T2b), against a margin of only **0.010** on the linear target. "
    "A five- to six-fold larger advantage appears exactly where the target contains structure a linear model cannot represent, and collapses "
    "where it does not. The comparison has the sensitivity the objection questions."
)
new_t1_para = (
    "On T1, the product-of-experts attains the lowest NLL (0.0382), ahead of the gated head (0.0636, Δ vs logistic −0.010, Holm p < 10⁻⁴) "
    "and the logistic regression (0.0736); the quantum circuit is worst (0.1651, Δ vs logistic +0.092, Holm p < 10⁻⁴). "
    "The pairwise likelihood differences on T1 show the product-of-experts leading all arms: Δ vs logistic −0.035, Δ vs gated −0.025, "
    "and Δ vs quantum −0.127 (all Holm p < 10⁻⁴). On T2a and T2b the gated head improves on logistic regression by 0.057 and 0.060, "
    "against 0.010 on T1. Supplementary Table S4 reports all pairwise differences across all targets. The comparison has the sensitivity the objection questions."
)
assert old_t1_para in text, "old_t1_para not found"
text = text.replace(old_t1_para, new_t1_para, 1)
edits.append(("E5 §2.4 T1 leadership and pairwise deltas", old_t1_para, new_t1_para, "User prompt E5"))

# --- E6: §2.5 (old §2.4) Ablation ---
old_abl_text = (
    "Repeating the ablation on the seven informative channels retires two of the three original findings and confirms the third. "
    "The likelihood penalty survives at half its magnitude and full significance: mean Δ = **+0.025** (95% CI +0.013 to +0.037), "
    "worse in **ten of ten seeds**, Holm *p* = 0.006. The accuracy penalty (*p* = 0.29) and the calibration benefit (*p* = 0.43) both disappear. "
    "A proper-score decomposition of the effect was non-identifiable at *n* = 447, so the mechanism is retired rather than replaced."
)
new_abl_text = (
    "Repeating the ablation on the seven informative channels retires two of the three original findings and confirms the third. "
    "The likelihood penalty survives at half its magnitude and full significance: mean Δ = **+0.025** (95% CI +0.013 to +0.037), "
    "worse in **ten of ten seeds**, Holm *p* = 0.006. The ten seeds vary initialisation on a fixed test set; the p-value therefore reflects "
    "optimiser variability, not sampling uncertainty over studies. We additionally report: "
    "(a) a patient-level bootstrap over test studies, 1,000 resamples: [PENDING: patient-level bootstrap point delta, SE, and 95% CI]; "
    "(b) [PENDING: K] repeated patient-disjoint re-splits, retraining each: [PENDING: mean delta across splits, between-split variance]. "
    "The ablation uses RY(πx) encoding without data re-uploading, whose output is restricted to a low-degree Fourier series in the inputs; "
    "it therefore characterises this encoding family only. To address this, we repeated the ablation with a data re-uploading ansatz at 8 qubits, "
    "[PENDING: L] layers, parameter count matched between entangled and product arms, ten seeds: [PENDING: re-uploading entangled vs product NLL and delta]. "
    "The lower training likelihood of the quantum circuit relative to the 54-parameter logistic arm (1.2598 vs 1.2142, Table 2) is consistent with "
    "either an unfavourable inductive bias or incomplete optimisation; our design cannot separate them. "
    "The accuracy penalty (*p* = 0.29) and the calibration benefit (*p* = 0.43) both disappear. "
    "A proper-score decomposition of the effect was non-identifiable at *n* = 447, so the mechanism is retired rather than replaced."
)
assert old_abl_text in text, "old_abl_text not found"
text = text.replace(old_abl_text, new_abl_text, 1)
edits.append(("E6 §2.5 Ablation seed variance, RY(pi x) expressivity, and pending slots", old_abl_text, new_abl_text, "User prompt E6"))

# --- E7: §2.6 (old §2.5) Sensitivity across 48 architectural configurations ---
old_sweep_section = text[text.index("### 2.6 Sensitivity across 48 architectural configurations"):text.index("![Figure 3]")]
new_sweep_section = (
    "### 2.6 Sensitivity across 48 architectural configurations\n\n"
    "A 48-cell sweep over qubit count {4, 6, 8}, depth 1–4, and entangling topology {none, linear, ring, full} was run with one split, one seed, and one training function, "
    "with trainable parameter count identical across topologies at fixed width and depth (`design_sweep.json`).\n\n"
    "The best cell by negative log-likelihood has zero entanglement (8 qubits, 1 layer, no entangler; NLL 1.1996, accuracy 0.5749), and three of the top five by NLL have zero entanglement. "
    "This does not support the served design, but because this is one split at one seed, selected from 48 cells, at 8 qubits (constant-channel confound of §2.5), it cannot establish that zero entanglement is optimal.\n\n"
)
text = text.replace(old_sweep_section, new_sweep_section, 1)
edits.append(("E7 §2.6 Sweep heading and reframing", old_sweep_section, new_sweep_section, "User prompt E7"))

# --- E8: §2.7 (old §2.6) BraTS Kernel comparison ---
old_brats_section = text[text.index("### 2.7 The quantum kernel does not differ from a classical kernel"):text.index("### 2.8 External validation falsifies one of our own hypotheses")]
new_brats_section = (
    "### 2.7 The quantum kernel does not differ from a classical kernel\n\n"
    "A 10-qubit fidelity kernel was compared against a radial basis function (RBF) kernel for glioma histological grade classification from brain-MRI encoder embeddings.\n\n"
    "**Held-out cohort (primary benchmark).** We lead with the clean evaluation on the 111 subjects held out from encoder training (55 validation + 56 test subjects whose segmentations never influenced any weight; `qkl_brats2020_official_heldout111.json`). "
    "This cohort comprises 23 Lower-Grade Glioma (LGG) and 88 High-Grade Glioma (HGG) subjects. Because the LGG count (23) is small, this analysis is statistically underpowered. "
    "On this held-out cohort, the quantum kernel's AUROC was 0.0217 below an RBF kernel: quantum AUROC **0.9407** against RBF **0.9625**, Δ = **−0.0217**, 95% CI −0.0543 to +0.0109, DeLong *p* = 0.191 (fold-level Wilcoxon p = 1.000; subject-level bootstrap median −0.0197, 95% CI −0.0577 to +0.0063 across 2,000 resamples). "
    "The confidence interval does not exclude a difference in either direction and admits an AUROC deficit of up to 0.054 for the quantum kernel. The pooled out-of-fold DeLong test should be read as approximate because Platt scaling parameters differ across folds.\n\n"
    "**Exploratory 369-subject cohort.** The 369-subject nested cross-validation across all official BraTS 2020 subjects (`qkl_brats2020_official.json`; 76 LGG, 293 HGG) is exploratory: "
    "the feature encoder was trained on roughly 258 of these subjects, so embeddings for those subjects are not strictly independent of the labels used in grading. "
    "Under 5-fold nested subject-level cross-validation with symmetric 5 × 5 grids, the quantum kernel attains subject-level AUROC **0.9422** against the RBF kernel's **0.9405**: "
    "Δ = **+0.0017**, 95% CI −0.0099 to +0.0132, DeLong *p* = 0.777 (subject-level bootstrap median +0.0016, 95% CI −0.0107 to +0.0126; fold-level Wilcoxon *p* = 0.875). Accuracy is 0.867 versus 0.865 and F1 is 0.723 versus 0.719.\n\n"
    "Both models show poor probability calibration at the subject level (ECE 0.308 and 0.284), which is an inherent artifact of averaging slice probabilities within a subject rather than a defect of either kernel.\n\n"
)
text = text.replace(old_brats_section, new_brats_section, 1)
edits.append(("E8 §2.7 BraTS reorder, 111-heldout primary (23 LGG / 88 HGG), DeLong p=0.777", old_brats_section, new_brats_section, "User prompt E8"))

# --- E10: Table 4 (old Table 3) Verdicts ---
t4_start = text.index("**Table 4.** The six claims, and what each survived.")
t4_end = text.index("---", t4_start)
old_t4_str = text[t4_start:t4_end].strip()

new_t4_str = (
    "**Table 4.** The six claims, diagnostic mechanisms, and final audit outcomes.\n\n"
    "| # | Claim | Diagnostic mechanism | Outcome |\n"
    "|---|---|---|---|\n"
    "| 1 | Better calibrated | Symmetric evaluation of post-processing calibration (P1) | **Withdrawn** (harness confound) |\n"
    "| 2 | Beats classical on likelihood | Standardised Adam optimiser protocol across all backends | **Withdrawn**; no difference detected under matched optimiser |\n"
    "| 3 | Entanglement helps | Capacity-matched 10-seed ablation, 7 informative channels | **Not supported**; worse NLL in 10/10 seeds (Holm *p* = 0.006, seed-level) |\n"
    "| 4 | Served design is good | 48-cell sweep across width, depth, and topology | **Not supported** (single-split sweep) |\n"
    "| 5 | Kernel beats classical kernel | Clean held-out 111-subject imaging benchmark | **Not supported**; 111-subject Δ −0.022 [−0.054, +0.011] |\n"
    "| 6 | Abstention typing by shot budget | Source-level code and measurement protocol audit (P7) | **Retracted** on source audit (analytic expectation plus synthetic Gaussian noise; never executes at finite shots) |"
)
text = text.replace(old_t4_str, new_t4_str, 1)
edits.append(("E10 Table 4 Verdicts", old_t4_str, new_t4_str, "User prompt E10"))

# --- E12: Discussion §3.2 mathematical failure softening ---
old_math_fail = (
    "Under exact statevector evaluation, the failure of quantum models to beat classical models cannot be blamed on the limitations of NISQ hardware. "
    "Instead, our findings prove that the failure is mathematical: as established by Huang et al. [10], Kübler et al. [11], and Thanasilp et al. [12], "
    "quantum feature maps in low dimensions suffer from unfavorable inductive biases, geometric concentration of measure, and rapid convergence toward classical kernels under hyperparameter tuning."
)
new_math_fail = (
    "Under exact statevector evaluation, the failure of quantum models to beat classical models cannot be blamed on the limitations of NISQ hardware. "
    "Instead, our results are consistent with the arguments of [10–12], which we did not test directly: "
    "quantum feature maps in low dimensions can suffer from unfavorable inductive biases, geometric concentration of measure, and rapid convergence toward classical kernels under hyperparameter tuning."
)
assert old_math_fail in text, "old_math_fail not found"
text = text.replace(old_math_fail, new_math_fail, 1)
edits.append(("E12 Discussion §3.2 mathematical softening", old_math_fail, new_math_fail, "User prompt E12"))

old_32_80 = "Over 80% of peer-reviewed papers claiming \"quantum advantage\" or \"quantum supremacy\" in digital health and medical imaging operate specifically within the 4–12 qubit regime [1]."
new_32_80 = "[PENDING: count of qualifying studies in our 231-record PubMed screen using 4–12 qubits, from screening log]."
assert old_32_80 in text, "old_32_80 not found"
text = text.replace(old_32_80, new_32_80, 1)
edits.append(("E3 §3.2 >80% claim removal", old_32_80, new_32_80, "User prompt E3"))

# --- E12 & E13: Discussion §3.3 Credentialism deletion, keeping recommendations & adding P6/D14 ---
sec33_start = text.index("### 3.3 Independent governance, defect transparency, and the continuous-integration standard")
sec33_end = text.index("## 4. Methods")
old_sec33_text = text[sec33_start:sec33_end]

new_sec33_text = (
    "### 3.3 Governance, verification limits, and empirical recommendations\n\n"
    "Protocol P6 guarantees that manuscript numbers equal the released artifacts. "
    "It does not guarantee the artifacts are correct: defect D14, in which real-trained models were evaluated on synthetic evidence while a validated verdict was printed, passed the drift test. "
    "We therefore noted the necessity of input-provenance assertions: [PENDING: input-provenance assertion to be implemented and committed before evaluation].\n\n"
    "What we leave standing is narrow and we state it as such. The quantum arm is the best-calibrated backend in this system under matched optimisation (§2.3), consistently across the paper. "
    "That is a real observation about the ansatz's inductive bias. It is not an advantage: it does not survive into accuracy, likelihood, or discrimination, it does not transfer, "
    "and it is available more cheaply from any arm by temperature scaling.\n\n"
    "For the field, we draw three recommendations. Calibrate every arm before comparing any of them, and report the temperature each required — the spread bounds the confound available. "
    "Fit every arm to convergence under a common budget, and report the residual gap on an arm whose optimum is knowable. "
    "And audit capability claims at source: our own principal contribution was invalidated by twelve lines of code that no metric could have contradicted.\n\n"
)
text = text.replace(old_sec33_text, new_sec33_text, 1)
edits.append(("E12 & E13 §3.3 Credentialism cut, P6/D14 limit, recommendations kept", old_sec33_text, new_sec33_text, "User prompt E12 & E13"))

# --- E14: Captions cleanup & commit hashes / test filenames ---
text = text.replace(" (commit `fa8c2b6`)", "")
text = text.replace("`docs/PREREGISTRATION_CHEXPERT.md`, commit `fa8c2b6`; ", "`docs/PREREGISTRATION_CHEXPERT.md`; ")
text = text.replace(" — the filename retains its v4 numbering", "")
text = text.replace("— the filename retains its v4 numbering", "")
text = text.replace("`export_brats2020_embeddings.py`", "the embedding pipeline")
text = text.replace("(`docs/figures/fig1_comparator_and_control.{png,pdf}`, generated by `aura/ml/evaluation/make_figures.py`.)", "")
text = text.replace("(`docs/figures/fig1_multiseed_ablation.{png,pdf}`.)", "")
text = text.replace("(`docs/figures/fig2_design_sweep.{png,pdf}`.)", "")
text = text.replace("(`docs/figures/fig5_external_validation.{png,pdf}`.)", "")
text = text.replace("(`aura/tests/test_doc_numbers.py`)", "")

# Cross-reference updates in Methods & text:
text = text.replace("(§2.1–§2.5, §2.7)", "(§2.2–§2.6, §2.8)")
text = text.replace("(§2.6)", "(§2.7)")
text = text.replace("(§2.9)", "(§2.1)")
text = text.replace("(§2.8)", "(§2.9)")
text = text.replace("(§2.2)", "(§2.3)")
text = text.replace("(§2.3)", "(§2.4)")
text = text.replace("(§2.4)", "(§2.5)")
text = text.replace("(§2.5)", "(§2.6)")

# Write MANUSCRIPT_v5.1.md
out_docs = repo_root / "docs" / "MANUSCRIPT_v5.1.md"
out_docs.write_text(text, encoding="utf-8")
out_root = Path("e:/AURA/MANUSCRIPT_v5.1.md")
out_root.write_text(text, encoding="utf-8")
print(f"Wrote MANUSCRIPT_v5.1.md successfully ({len(text)} chars).")

# Write CHANGELOG.md
changelog_lines = [
    "# CHANGELOG: MANUSCRIPT_v5.0.md -> MANUSCRIPT_v5.1.md\n",
    "Detailed record of every edit applied to `docs/MANUSCRIPT_v5.0.md` to produce `MANUSCRIPT_v5.1.md`.\n"
]
for edit_name, old_c, new_c, src in edits:
    changelog_lines.append(f"## {edit_name}\n")
    changelog_lines.append(f"**Source / Justification:** {src}\n")
    changelog_lines.append("```diff\n- " + old_c.strip().replace("\n", "\n- ") + "\n+ " + new_c.strip().replace("\n", "\n+ ") + "\n```\n")

(repo_root / "docs" / "CHANGELOG.md").write_text("\n".join(changelog_lines), encoding="utf-8")
Path("e:/AURA/CHANGELOG.md").write_text("\n".join(changelog_lines), encoding="utf-8")
print("Wrote CHANGELOG.md successfully.")

# Write PENDING_LIST.md
pending_list = r"""# PENDING_LIST: Outstanding Runs and Lookups for MANUSCRIPT_v5.1.md

Every `[PENDING]` marker in `MANUSCRIPT_v5.1.md` is cataloged below with the exact computation, run, or database lookup required to resolve it. No placeholder may be filled without a verified artifact on disk.

---

### 1. Qubit Distribution from PubMed Screening Log
- **Location:** Introduction (p. 2) and Section 3.2 (p. 11)
- **Placeholder:** `[PENDING: count of qualifying studies in our 231-record PubMed screen using 4–12 qubits, from screening log]`
- **Required Action:** Run `audit_screen_qubits.py` over the 231 screened records in `bench/field_audit/` to extract:
  1. Number of qualifying empirical studies with explicit qubit counts.
  2. Number and percentage of studies operating within the 4–12 qubit regime.
  3. Screening protocol reference.
- **Contingency Rule:** If this tally cannot be verified against the screening log, drop the "urgency" argument entirely and justify the 8–10 qubit regime on noise-free classical simulability alone.

---

### 2. Field Audit Screening Flow Funnel Counts
- **Location:** Section 2.1 (Field-wide audit, p. 3)
- **Placeholder:** `[PENDING: screening flow 231 -> eligible -> reproducible -> 5 with counts from the screening log]`
- **Required Action:** Extract exact breakdown counts from the screening log:
  1. Exclusions at Title/Abstract screening (conceptual proposals, non-clinical simulations, perspectives, non-health data).
  2. Exclusions at Full-Text screening (closed/proprietary data, missing source code/hyperparameters, under-specified circuits, non-deterministic execution).
  3. Yielding the 5 reproducible audited studies.

---

### 3. Pre-registered TOST Equivalence Test for Likelihood Parity
- **Location:** Section 2.3 (old §2.2, p. 5)
- **Placeholder:** `[PENDING: margin to be pre-registered in an addendum before execution, e.g. 0.01 NLL]: [PENDING result]`
- **Required Action:** 
  1. Deposit a pre-registration addendum (`PREREGISTRATION_ADDENDUM.md`) establishing the equivalence margin $\Delta_{\text{equiv}} = \pm 0.010$ NLL prior to analyzing test outcomes.
  2. Execute `tost_equivalence.py` on the paired test likelihood differences between Quantum VQC and Product-of-Experts from `fusion_matched_optimiser_predictions.npz`.
  3. Report two one-sided $t$-statistics, $p_{\text{TOST}}$, and the 90% confidence interval.
- **Contingency Rule:** Until the pre-registration and test results are reported, maintain that no difference was detected; do not claim parity.

---

### 4. Patient-Level Clustered Bootstrap on Test Studies
- **Location:** Section 2.5 (old §2.4, p. 7)
- **Placeholder:** `[PENDING: patient-level bootstrap point delta, SE, and 95% CI]`
- **Required Action:** 
  1. Execute `ablation_patient_bootstrap.py` using clustered resampling over unique patient identifiers across the 447 MIMIC-CXR test studies (1,000 resamples).
  2. Compute bootstrap standard error and 95% percentile confidence interval for $\Delta\text{NLL} = \text{NLL}_{\text{entangled}} - \text{NLL}_{\text{product}}$.

---

### 5. Repeated Patient-Disjoint Re-splits for the Ablation
- **Location:** Section 2.5 (old §2.4, p. 7)
- **Placeholder:** `[PENDING: K] repeated patient-disjoint re-splits, retraining each: [PENDING: mean delta across splits, between-split variance]`
- **Required Action:**
  1. Perform $K = 5$ repeated patient-disjoint stratified splits (70/10/20 train/cal/test).
  2. Retrain both entangled and unentangled backends under the matched Adam protocol on each split.
  3. Calculate empirical between-split variance ($\sigma^2_{\text{data}}$) alongside optimizer initialization variance ($\sigma^2_{\text{init}}$).

---

### 6. Matched-Parameter Data Re-uploading Circuit Ablation
- **Location:** Section 2.5 (old §2.4, p. 7)
- **Placeholder:** `[PENDING: L] layers, parameter count matched between entangled and product arms, ten seeds: [PENDING: re-uploading entangled vs product NLL and delta]`
- **Required Action:**
  1. Execute `reuploading_ablation.py` at 8 qubits (7 informative channels + 1 reference) across $L=3$ layers with alternating data re-uploading blocks.
  2. Strictly match parameter count between entangled and product arms (72 trainable parameters each).
  3. Train across 10 random seeds under the matched Adam protocol.
  4. Report paired $\Delta\text{NLL}$, confidence interval, and Holm $p$-value.

---

### 7. Input-Provenance Cryptographic Assertion Protocol (Defect D14 Resolution)
- **Location:** Section 3.3 (Governance and verification limits, p. 11)
- **Placeholder:** `[PENDING: input-provenance assertion to be implemented and committed before evaluation]`
- **Required Action:**
  1. Commit `provenance_assertion.py` to the repository.
  2. Integrate SHA-256 byte-digest assertions over raw evidence tensors into the CI test suite to guarantee that models are never evaluated against synthetic test stubs without triggering an automated build failure.
"""

(repo_root / "docs" / "PENDING_LIST.md").write_text(pending_list, encoding="utf-8")
Path("e:/AURA/PENDING_LIST.md").write_text(pending_list, encoding="utf-8")
print("Wrote PENDING_LIST.md successfully.")
