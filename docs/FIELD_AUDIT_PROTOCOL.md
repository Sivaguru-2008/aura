# Field audit protocol: do published quantum-advantage claims in medical machine learning survive a symmetric comparison?

**Fixed** 21 September 2026, after the literature search and title/abstract listing, before any full text was read for eligibility and before any model was run.
**Author:** Sivaguru R. M.

## 1. Question

Among peer-reviewed studies that apply a quantum machine-learning model to a medical prediction task and report that it outperforms classical models, what fraction of those claims survive when the quantum model and strong classical models are evaluated on the same data under one symmetric protocol?

## 2. Sampling frame

- **Database.** PubMed, searched 21 September 2026 through the NCBI E-utilities interface.
- **Query.** `("quantum machine learning"[Title/Abstract] OR "variational quantum classifier"[Title/Abstract] OR "quantum support vector"[Title/Abstract] OR "quantum kernel"[Title/Abstract] OR "quantum neural network"[Title/Abstract]) AND (classification[Title/Abstract] OR diagnosis[Title/Abstract] OR prediction[Title/Abstract])`, publication dates 2019-01-01 to 2026-08-31.
- **Records.** 231 (PMID list: `bench/field_audit/search/all_pmids.txt`).

PubMed is chosen because it is the index of record for biomedical literature and its queries are exactly reproducible. It does not index arXiv, IEEE Xplore or ACM; studies published only there are outside the frame. This is a stated limitation, not a correction to be applied after the results.

## 3. Screening

| Stage | Rule | Applied by |
|---|---|---|
| S1 | Title or abstract matches a medical/health term list, and the record is not a review, survey or perspective (regex recorded in `bench/field_audit/search/screen.py`) | script |
| S2 | Title/abstract: a primary study predicting a clinical or physiological outcome at the level of a human patient or subject (diagnosis, grade, prognosis, phenotype) — molecular, peptide, protein, compound and cell-line prediction tasks are excluded as cheminformatics/bioinformatics; at least one quantum model (circuit-based: variational classifier, quantum kernel, quantum neural network, quanvolution) is evaluated; the study reports that a quantum model outperforms at least one classical model on a stated performance metric. "Quantum-inspired" methods that run no quantum circuit are excluded. Chemistry "QML" (quantum-chemistry machine learning) is excluded. | author |
| S3 | Full text: (a) the data are downloadable without credentialed access, registration-gated approval, or request to the authors; (b) the input to the quantum model is a feature vector — tabular data, or features the paper derives by a reproducible pipeline — so that the quantum model itself can be simulated exactly; (c) the quantum model is specified well enough to implement (encoding, qubit count, ansatz or kernel, training procedure), or code is public; (d) the superiority claim is quantitative. | author |

For S3(a), data behind any account sign-in (including Kaggle) count as not openly downloadable: the auditor does not create or use accounts. A record whose full text cannot be reached without a publisher login, and has no open preprint, is reported as **not assessable** — a property of access, not of the study — rather than excluded on a criterion.

Every excluded record gets one recorded reason. Hybrid deep-learning models in which a quantum layer sits inside an image CNN are excluded at S3(b) in this round: the claim there concerns a large classical network with a small quantum head, and testing it requires the image pipeline and a quantum-layer ablation — a different experiment, listed as future work.

## 4. For each included study, two arms

### 4.1 Reproduction arm

Implement the study's quantum model and its reported classical comparator(s) following its described protocol (split, preprocessing, feature count, qubits, encoding, ansatz/kernel, optimiser, epochs) as closely as the paper allows. Where the paper is silent, the choice is recorded. Where code is public it is read, never executed; our implementation follows it.

**Fidelity.** The reproduction is *faithful* if the reproduced quantum metric is within 5 percentage points (accuracy, F1) or 0.05 (AUROC) of the reported value. Unfaithful reproductions are reported, not dropped, and the symmetric-arm verdict for that study is flagged.

**Source audit.** Where code is available, it is checked for the leakage patterns catalogued by Kapoor and Narayanan: preprocessing, scaling, feature selection or resampling (e.g. SMOTE) fitted before the train/test split; test data used for model or epoch selection; duplicate rows crossing the split.

### 4.2 Symmetric arm

All arms see identical data, identical preprocessing (the study's own feature pipeline, fitted inside each training fold), and identical folds.

- **Evaluation.** Repeated stratified 5-fold cross-validation, 5 repeats (25 outer folds), seed 7. Subject- or patient-level grouping wherever the data contain repeated measurements.
- **Tuning.** Nested: every arm's hyperparameters are chosen by 3-fold inner cross-validation on the outer training fold, from a grid of 9–16 configurations. No arm gets a larger grid than the quantum arm's.
- **Quantum arm.** The study's quantum model. Its grid covers the quantity that governs its inductive bias — the feature-map bandwidth for kernels (Shaydulin and Wild); learning rate × layers for variational models.
- **Classical arm.** One arm: *nested-selected classical*. Inside each outer training fold, inner cross-validation chooses among logistic regression, RBF-SVM, random forest, histogram gradient boosting and an MLP (each with its own small grid). This is a single pre-specified comparator, so there is no multiplicity and no selection on test folds.
- **The study's own named comparator** is also run under the symmetric protocol and reported descriptively.
- **Grid budget.** The quantum arm and the nested-selected classical arm each search 16 configurations in total (classical: logistic regression 3, RBF-SVM 4, random forest 3, gradient boosting 3, MLP 3).
- **Selection metric.** Inner-CV selection uses the primary metric when it is accuracy, AUROC or F1. When the claim is a single-class rate (sensitivity, specificity, recall), selection uses balanced accuracy — a single-class rate is maximised trivially by predicting one class — and the claimed rate is then evaluated at the model's own 0.5 threshold.
- **Sample-size claims.** When a claim is about a reduced training set, each outer training fold is reduced to a stratified random subset of the stated fraction (the same subset for every arm). The smallest fraction the study reports numbers for is primary; larger fractions it names are secondary.
- **Class balance.** When the claim is a single-class rate and the study's quantum model uses class-balanced weighting, every arm uses class-balanced weighting (`class_weight='balanced'`; the MLP is trained on a minority-oversampled copy of its training fold).
- **Metrics.** Primary: the metric on which the study claims superiority. Secondary: AUROC and negative log-likelihood, with probabilities Platt-calibrated on the inner folds for every arm that does not output probabilities.

### 4.3 Verdict, per claim

The fold-level difference (quantum − nested-selected classical) on the primary metric is tested with the Nadeau–Bengio corrected resampled t-test (25 folds, test/train ratio 0.25), two-sided.

- **Survives:** quantum significantly better (p < 0.05).
- **Not supported:** no significant difference.
- **Reversed:** classical significantly better.

The headline result is the count of claims in each category. No study is removed after its result is known.

## 5. What is not claimed

The audit tests the specific superiority claims in the frame. It does not test whether quantum models *can* ever help on medical data, and a "not supported" verdict does not mean the study is wrong about anything else it reports.
