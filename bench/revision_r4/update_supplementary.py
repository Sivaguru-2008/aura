import re
from pathlib import Path

supp_path = Path("supplementary_information.tex")
text = supp_path.read_text(encoding="utf-8")

# 1. Update Title
old_title = r"{\LARGE\textbf{\textsf{Claimed quantum benefits in clinical machine learning do not survive symmetric evaluation}}}\\[12pt]"
new_title = r"{\LARGE\textbf{\textsf{Claimed quantum benefits in clinical machine learning could not be verified under symmetric evaluation: an empirical and literature audit}}}\\[12pt]"

if old_title in text:
    text = text.replace(old_title, new_title)
    print("Supplementary title updated.")
else:
    print("WARNING: old_title not found.")

# 2. Add PRISMA literature audit & single-rater disclosure to Supplementary Note 5
old_s5_header = r"\section{Supplementary Note 5: Full 17-Item Audit Scoring Matrix for Eligible Literature Studies}"
new_s5_section = r"""\section{Supplementary Note 5: Systematic Literature Search, PRISMA Flow, and 17-Item Audit Scoring Matrix}

\subsection{Literature Search Strategy and PRISMA Flow Accounting}
To ensure complete transparency and independent auditability, our systematic literature screening followed pre-specified inclusion and exclusion criteria documented at commit \texttt{61e777b}:
\begin{itemize}
  \setlength{\itemsep}{1pt}
  \item \textbf{Search Query:} \texttt{("quantum machine learning" OR "variational quantum") AND ("clinical" OR "medical")}
  \item \textbf{Database Searched:} PubMed / MEDLINE (National Library of Medicine).
  \item \textbf{Search Execution Date:} September 21, 2026.
  \item \textbf{Total Records Retrieved:} $N = 231$ records (documented with PMIDs, titles, and abstracts in repository file \texttt{bench/field\_audit/search/pubmed\_records.json}).
\end{itemize}

The systematic screening progression proceeded through three pre-specified stages:
\begin{enumerate}
  \setlength{\itemsep}{2pt}
  \item \textbf{Stage S1 (Medical Relevance and Review Exclusion):} Title and abstract screening for human medical/health applications using automated regex filtering (\texttt{bench/field\_audit/search/screen.py}). Records focusing on theoretical physics, non-human benchmarks, materials science, or published reviews/surveys were excluded ($n = 102$). Exactly \textbf{129 records} were retained (\texttt{stage1\_medical.json}).
  \item \textbf{Stage S2 (Empirical Advantage Claim Adjudication):} Full abstract screening of all 129 medical records for explicit claims of empirical quantum-over-classical performance superiority on a clinical outcome. Records were excluded ($n = 88$) if they reported no empirical superiority, found classical models superior or equivalent, evaluated purely molecular/compound/docking targets without human patients, or evaluated general non-medical image benchmarks. Exactly \textbf{41 records} claimed an empirical quantum advantage on clinical outcomes (\texttt{s2\_decisions.csv}).
  \item \textbf{Stage S3 (Eligibility and Reconstructibility Adjudication):} Full-text assessment of all 41 candidate studies against data and code accessibility criteria. Studies were excluded ($n = 33$) due to: hybrid deep networks where quantum circuits were embedded inside massive convolutional neural networks on raw images without released code or checkpoints ($n = 18$); private/undisclosed hospital cohorts ($n = 7$); full-text inaccessible behind publisher login ($n = 4$); unspecified quantum pipeline ($n = 2$); classical neural network mislabeled as quantum ($n = 1$); or advantage claimed only between quantum variants ($n = 1$). Exactly \textbf{8 publications} met eligibility criteria (\texttt{s3\_decisions.csv}).
  \item \textbf{Reconstructibility Adjudication:} Of the 8 eligible publications, 3 were non-reconstructible: Jahin et al. (PMID 40865268, 3M parameters, no code released); Chen et al. (PMID 42046760, proprietary MRI radiomics features); and Choi et al. (PMID 41966795, controlled-access omics data). Exactly \textbf{5 publications (spanning 6 clinical prediction tasks)} were fully reconstructible and computationally reimplemented.
\end{enumerate}

\subsection{Operational Definitions of the 17-Item Checklist and Single-Rater Disclosure}
\noindent\textbf{Single-Rater Scoring Limitation Disclosure:} In adherence to rigorous scientific reporting standards, we explicitly disclose that the 17-item checklist was scored by a single reviewer (S.R.M.). No second independent reviewer scoring exists; to maintain absolute scientific integrity, no artificial inter-rater statistics or manufactured second-reviewer scores are reported. To eliminate subjectivity and provide an auditable basis for independent peer review, every checklist item is operationalized via strict, binary decision rules:
\begin{itemize}
  \setlength{\itemsep}{1pt}
  \item \textbf{Item 1 (Independent Calibration):} YES if predicted probabilities are calibrated separately per model arm using dedicated validation data; NO if models are evaluated using raw, unscaled logits or if only one arm is scaled.
  \item \textbf{Item 2 (Validation-Only Calibration):} YES if calibration parameters (e.g. temperature $T$) are fitted strictly without test-set access; NO if test labels/features are accessed during calibration.
  \item \textbf{Item 3 (Calibration Metrics Reported):} YES if probability calibration metrics (ECE, Brier score) are reported alongside discrimination metrics (AUROC); NO if only thresholded accuracy is reported.
  \item \textbf{Item 4 (Matched Tuning Budget):} YES if classical baselines receive an equivalent or greater number of hyperparameter search trials across standard model families; NO if classical baselines are evaluated under default, untuned settings.
  \item \textbf{Item 5 (Nested CV / Isolated Selection):} YES if model hyperparameters are selected via nested cross-validation or isolated validation splits; NO if hyperparameters are tuned on the test set.
  \item \textbf{Item 6 (Competitive Nonlinear Baselines):} YES if classical comparators include standard nonlinear model families (GBDT, Random Forests, SVM-RBF, MLPs); NO if compared only against deliberately restricted linear models.
  \item \textbf{Item 7 (Explicit Architecture Parameters):} YES if parameter counts, qubit counts, circuit depth, and optimizer stopping conditions are documented; NO if unspecified.
  \item \textbf{Item 8 (Patient-Disjoint Partitioning):} YES if dataset partitions guarantee zero patient overlap ($|S_{\text{train}} \cap S_{\text{test}}| = 0$); NO if patient records cross partition boundaries.
  \item \textbf{Item 9 (Grouped Repeated Measures):} YES if cross-validation groups repeated measurements (temporal EEG epochs, repeated phonations) by subject identifier; NO if repeated recordings are split randomly as independent instances; N/A if dataset contains strictly one record per subject.
  \item \textbf{Item 10 (In-Fold Feature Preprocessing):} YES if dimensionality reduction (PCA), standardization, and imputation are fitted strictly on training folds; NO if fitted on pooled data prior to partitioning.
  \item \textbf{Item 11 (Paired Inferential Testing):} YES if superiority claims are evaluated using paired hypothesis tests or uncertainty intervals on performance differences; NO if asserted from point estimates alone.
  \item \textbf{Item 12 (Dependency Correction):} YES if repeated cross-validation incorporates dependency corrections (e.g. Nadeau--Bengio) accounting for outer-fold data overlap; NO if conventional paired $t$-tests treat overlapping CV folds as independent.
  \item \textbf{Item 13 (Multiplicity Adjustment):} YES if multiplicity corrections (Holm--Bonferroni) are applied across the family of pairwise model comparisons; NO if unadjusted.
  \item \textbf{Item 14 (Formal Equivalence Testing):} YES if equivalence claims are substantiated via Two One-Sided Tests against a pre-specified margin; NO if equivalence is asserted from failure to reject the null.
  \item \textbf{Item 15 (Complete Outcome Reporting):} YES if all evaluated comparator arms and negative results are disclosed; NO if unfavorable baselines are selectively omitted.
  \item \textbf{Item 16 (Immutable Derived Artifacts):} YES if every reported metric corresponds to a public, reproducible on-disk artifact with cryptographic hashes; NO if numbers exist only in manuscript text without verifiable code checkpoints.
  \item \textbf{Item 17 (Mechanistic Code Verification):} YES if claimed quantum physical mechanisms (e.g. entanglement) are verified through ablation experiments demonstrating causal necessity; NO if claimed without empirical mechanism testing.
\end{itemize}"""

if old_s5_header in text:
    text = text.replace(old_s5_header, new_s5_section)
    print("Supplementary Note 5 updated with PRISMA flow and operational definitions.")
else:
    print("WARNING: old_s5_header not found.")

# 3. Add Clinical Metrics & Confusion Matrices to Supplementary Note 9
old_s9_end = r"\end{table}"
# We append a dedicated subsection to Supplementary Note 9
supp_addition = r"""
\subsection{Detailed Condition-Specific Clinical Performance Metrics and Confusion Matrices}
To supplement the log-likelihood decomposition in Table 4, Table S5 reports comprehensive condition-specific classification metrics across all five active clinical triage categories evaluated on the 56,534 held-out patients under the pre-registered argmax triage decision rule:

\begin{table}[h]
\centering
\small
\caption*{\textbf{\textsf{Supplementary Table 5 \textbar\ Detailed condition-specific clinical performance metrics across 56,534 held-out patients.}}}
\begin{tabular}{lcccccc}
\toprule
\textbf{Condition \& Prevalence} & \textbf{Model Backend} & \textbf{Sensitivity (Recall)} & \textbf{Specificity} & \textbf{Precision (PPV)} & \textbf{AUROC} & \textbf{AUPRC} \\
\midrule
\textbf{Normal} & Tuned VQC & 0.7738 & 0.7786 & 0.6142 & 0.8500 & 0.7085 \\
($n = 17,692$, 31.29\%) & Tuned PoE & \textbf{0.7841} & 0.7691 & 0.6073 & 0.8504 & 0.7080 \\
& Tuned Gated Head & 0.7681 & \textbf{0.7837} & \textbf{0.6179} & \textbf{0.8508} & \textbf{0.7084} \\
\addlinespace
\textbf{Pneumonia} & Tuned VQC & \textbf{0.5400} & 0.7128 & 0.4481 & 0.6922 & \textbf{0.4675} \\
($n = 17,052$, 30.16\%) & Tuned PoE & 0.4678 & \textbf{0.7676} & \textbf{0.4651} & \textbf{0.6937} & 0.4667 \\
& Tuned Gated Head & 0.5360 & 0.7132 & 0.4467 & 0.6936 & 0.4667 \\
\addlinespace
\textbf{Heart Failure} & Tuned VQC & 0.4391 & \textbf{0.8660} & \textbf{0.5744} & \textbf{0.7661} & 0.5590 \\
($n = 16,490$, 29.17\%) & Tuned PoE & \textbf{0.5111} & 0.8248 & 0.5457 & 0.7653 & 0.5584 \\
& Tuned Gated Head & 0.4560 & 0.8561 & 0.5661 & 0.7658 & \textbf{0.5611} \\
\addlinespace
\textbf{Pneumothorax} & Tuned VQC & \textbf{0.1223} & 0.9912 & 0.4476 & 0.7525 & \textbf{0.2318} \\
($n = 3,108$, 5.50\%) & Tuned PoE & 0.1162 & 0.9914 & 0.4402 & 0.7568 & 0.2303 \\
& Tuned Gated Head & 0.0911 & \textbf{0.9941} & \textbf{0.4717} & \textbf{0.7609} & 0.2303 \\
\addlinespace
\textbf{Malignancy} & Tuned VQC & 0.0474 & 0.9975 & 0.4315 & \textbf{0.7132} & 0.1425 \\
($n = 2,192$, 3.88\%) & Tuned PoE & \textbf{0.0529} & 0.9971 & 0.4203 & 0.6975 & 0.1424 \\
& Tuned Gated Head & 0.0447 & \textbf{0.9981} & \textbf{0.4900} & 0.7026 & \textbf{0.1448} \\
\bottomrule
\end{tabular}
\end{table}

\noindent\textbf{Confusion Matrix Analysis and False-Negative Patterns:} Under multi-class argmax triage assignment across the five active categories (ordered Normal, Pneumonia, Heart Failure, Malignancy, Pneumothorax):
\begin{itemize}
  \item \textbf{Tuned VQC Confusion Matrix ($N=56,534$):}
  \begin{equation*}
  C_{\text{VQC}} = \begin{pmatrix}
  13,690 & 2,614 & 1,270 & 26 & 92 \\
  4,117 & 9,208 & 3,447 & 58 & 222 \\
  2,774 & 6,311 & 7,241 & 27 & 137 \\
  796 & 1,049 & 225 & 104 & 18 \\
  912 & 1,367 & 423 & 26 & 380
  \end{pmatrix}
  \end{equation*}
  \item \textbf{Tuned PoE Confusion Matrix ($N=56,534$):}
  \begin{equation*}
  C_{\text{PoE}} = \begin{pmatrix}
  13,873 & 2,095 & 1,615 & 23 & 86 \\
  4,321 & 7,977 & 4,460 & 70 & 224 \\
  2,852 & 5,045 & 8,428 & 34 & 131 \\
  843 & 883 & 332 & 116 & 18 \\
  954 & 1,151 & 609 & 33 & 361
  \end{pmatrix}
  \end{equation*}
\end{itemize}
This confusion analysis reveals critical clinical insights:
\begin{enumerate}
  \item \textbf{Severe Minority-Class Suppression:} For Malignancy ($n=2,192$), both models correctly identify only $\approx 5\%$ of cases under unweighted argmax assignment (104 for VQC, 116 for PoE). Out of 2,192 true Malignancy cases, VQC misclassifies 796 as Normal and 1,049 as Pneumonia. Similarly, for Pneumothorax ($n=3,108$), VQC misclassifies 912 cases as Normal and 1,367 as Pneumonia.
  \item \textbf{Heart Failure Disparity:} Classical PoE correctly identifies \textbf{8,428 Heart Failure patients} (51.11\% sensitivity) compared to \textbf{7,241 for VQC} (43.91\% sensitivity), detecting 1,187 more acute heart failure cases than the quantum model.
  \item \textbf{Pneumonia Compensation:} VQC compensates for this heart failure loss by predicting Pneumonia more aggressively (9,208 true positives, 54.00\% sensitivity vs 7,977 for PoE, 46.78\% sensitivity).
\end{enumerate}
This demonstrates that population-level equivalence is achieved through opposite-direction sensitivity shifts on common conditions, while minority acute conditions (Malignancy and Pneumothorax) experience severe degradation that breaks class-balanced operational equivalence ($\Delta\text{NLL}_{\text{macro}} = +0.03892$ nats, $p_{\text{TOST}} > 0.99$).

\section{Supplementary References}
\begin{enumerate}
  \small
  \setlength{\itemsep}{1pt}
  \item Gupta, S. et al. Quantum machine learning in digital health: a systematic review. \textit{npj Digit. Med.} \textbf{8}, 42 (2025).
  \item Rendle, S. et al. Are we really making much progress? A worrying analysis of recent neural recommendation approaches. In \textit{Proc. RecSys} 101--109 (2019).
  \item Musgrave, K. et al. A metric learning reality check. In \textit{Proc. ECCV} 681--699 (2020).
  \item Glocker, B. et al. Machine learning for medical image analysis: are we making progress? \textit{Lancet Digit. Health} \textbf{4}, e680--e681 (2022).
  \item Huang, H.-Y. et al. Power of data in quantum machine learning. \textit{Nat. Commun.} \textbf{12}, 2631 (2021).
  \item Schuld, M., Sweke, R. \& Meyer, J. J. Effect of data encoding on the expressive power of variational quantum-machine-learning models. \textit{Phys. Rev. A} \textbf{103}, 032430 (2021).
  \item Nadeau, C. \& Bengio, Y. Inference for the generalization error. \textit{Mach. Learn.} \textbf{52}, 239--281 (2003).
  \item Schuirmann, D. J. A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. \textit{J. Pharmacokinet. Biopharm.} \textbf{15}, 657--680 (1987).
  \item Lakens, D. Equivalence testing for psychological research: a tutorial on TOST design, power, and statistical analyses. \textit{Soc. Psychol. Personal. Sci.} \textbf{8}, 355--362 (2017).
  \item Guo, C., Pleiss, G., Sun, Y. \& Weinberger, K. Q. On calibration of modern neural networks. In \textit{Proc. ICML} 1321--1330 (2017).
  \item Altman, D. G. \& Bland, J. M. Absence of evidence is not evidence of absence. \textit{BMJ} \textbf{311}, 485 (1995).
  \item Irvin, J. et al. CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison. In \textit{Proc. AAAI} 590--597 (2019).
  \item Johnson, A. E. W. et al. MIMIC-CXR-JPG, a large publicly available database of labeled chest radiographs. \textit{Sci. Data} \textbf{6}, 317 (2019).
  \item Parthasarathi, P. et al. Variational quantum classifier-based early identification and classification of chronic kidney disease using sparse autoencoder and LASSO shrinkage. \textit{PeerJ Comput. Sci.} \textbf{11}, e2789 (2025).
  \item Daga, M., Naole, S. et al. Adaptive quantum kernel selection via leakage-free stacking for clinical diagnostics on NISQ hardware. \textit{Sci. Rep.} \textbf{16}, 12841 (2026).
  \item Aksoy, G. \& Cattan, G. Quantum machine-based decision support system for the detection of schizophrenia from EEG records. \textit{J. Med. Syst.} \textbf{48}, 29 (2024).
  \item Akpinar, E. \& Oduncuoglu, M. Hybrid classical and quantum computing for enhanced glioma tumor classification using TCGA data. \textit{Sci. Rep.} \textbf{15}, 25935 (2025).
  \item Ghisoni, F., Borrotti, M. \& Mariani, P. A large scale statistical analysis of quantum and classical neural networks in the medical domain. \textit{Sci. Rep.} \textbf{16}, 3719 (2026).
\end{enumerate}
"""

# Place addition before \end{document}
if r"\end{document}" in text:
    text = text.replace(r"\end{document}", supp_addition + "\n\\end{document}")
    print("Supplementary Note 9 additions and references added.")
else:
    print("WARNING: \\end{document} not found.")

supp_path.write_text(text, encoding="utf-8")
print("Done writing supplementary_information.tex.")
