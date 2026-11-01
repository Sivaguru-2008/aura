# Symmetric Evaluation Checklist applied to the eight eligible studies

Analysis K of `PREREGISTRATION_v6_2026-10-03.md` §7. Items and decision rules are those of
`SYMMETRIC_EVALUATION_CHECKLIST.md`, committed (61e777b) before any study was scored.

## How the scoring was done

- **Rater.** Single rater: an AI assistant working for the author, scoring 7 October 2026. A second, independent human rater is sought; until then these are single-rater judgements.
- **Source text.** Full text was retrieved from PubMed Central for the six studies that have it. The EEG (38441727) and PROSTATEx (42046760) papers were read in full. For CKD, glioma, stacking and heart, the rater read every paragraph that matched the checklist's subject matter (splitting, preprocessing, tuning, stopping, statistics, code, calibration, mechanism), extracted by keyword. Passages that matched no keyword were not read.
- **Back matter.** PMC full-text bodies omit back matter such as data and code availability statements. C6.1 is therefore *Unclear* unless the body itself states availability.
- **No PMC text.** KACQ-DCNN (40865268) and INCOV (41966795) are scored only from the S3 full-text screening notes in `bench/field_audit/search/s3_decisions.csv`. Most of their items are *Unclear* for that reason.
- **Evidence column.** It paraphrases the deciding passage. "Re-test" refers to our field-audit result where one exists.

Answers: **Y** = Yes, **N** = No, **U** = Unclear, **–** = N/A.

## Summary

| Study (PMID) | C1.1 | C2.1 | C2.2 | C3.1 | C3.2 | C3.3 | C4.1 | C4.2 | C5.1 | C6.1 | C7.1 | At risk of asymmetric evaluation? | Unsupported by inference? | Re-test verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CKD VQC (40567645) | Y | N | N | U | U | U | N | N | U | U | N | **yes** (C2.1) | **yes** | REVERSED |
| Glioma VQC (40676161) | Y | N | U | Y | N | U | N | N | Y | U | N | **yes** (C2.1, C3.2) | **yes** | NOT SUPPORTED |
| Kernel stacking, PD/diabetes (42286086) | Y | N | N | N | N | Y | Y | N | Y | U | N | **yes** (C2.1, C3.1, C3.2) | no | NOT SUPPORTED ×2 |
| Heart QNN (41513819) | Y | Y | U | Y | U | N | N | N | Y | U | – | **yes** (C3.3) | **yes** | see H |
| Schizophrenia EEG QSVM (38441727) | Y | N | U | U | U | U | N | N | U | U | N | **yes** (C2.1) | **yes** | NOT SUPPORTED |
| KACQ-DCNN heart (40865268) | U | U | U | U | U | U | U | U | U | N (no code) | U | not assessable | not assessable | not run (no code) |
| PROSTATEx QSVM (42046760) | Y | Y | Y | Y | Y | Y | N | N | Y | U | – | no | **yes** | not run |
| INCOV omics QSVM (41966795) | U | U | U | U | U | U | U | U | U | U | U | not assessable | not assessable | not run |

C1.2, C2.3, C2.4, C4.3, C6.2 and C7.2 are given per study below. They change none of the summary labels.

**Reading.** Five of the six assessable studies are at risk of asymmetric evaluation under the
checklist's summary rule. Five of the six support the claimed superiority without a paired test
or interval on the difference. The one study that passes every item at risk (PROSTATEx) is also
the one whose own text says the quantum kernel only *matched* the strong classical comparators
(random forest, XGBoost). Its large margin is over an SVM-RBF with a fold-to-fold SD of 0.244.
These labels describe reporting and design. They do not show that a claim is false; the
field-audit re-tests address that.

## Per-study evidence

### CKD — PeerJ Comput Sci 2025, PMID 40567645
- **C1.1 Y.** No post-processing is described for any arm.
- **C2.1 N.** The VQC is presented as needing no intensive hyperparameter tuning, and no comparator search is described. The comparators are fixed classical models.
- **C2.2 N.** The comparators are weak or fixed (SVM, RF, LR, MLP, RBF network). No tuned boosting model is included.
- **C3.1 U.** One row per patient, so there are no repeated units, but data augmentation with a sparse autoencoder is applied. Whether the augmented rows can cross the split is not stated, beyond an unnamed holdout check of augmentation.
- **C3.2 U.** Min–max normalisation over the dataset and LASSO selection are described. Whether either is fitted inside the training data is not stated.
- **C3.3 U.**
- **C4.1 N.** Point accuracies only.
- **C4.2 N.**
- **C4.3 –.**
- **C5.1 U.**
- **C6.1 U.**
- **C7.1 N.** Entanglement and the "exponentially large feature space" are asserted as the reason for the result, with no ablation.
- **C7.2 –.**
- **Re-test:** the VQC reached 83.5% under the paper's own protocol (reported 99.2%). Classical models reached 96–98%. Symmetric verdict: REVERSED.

### Glioma grade — Sci Rep 2025, PMID 40676161
- **C1.1 Y.**
- **C2.1 N.** Six VQC variants with different feature maps, circuits and optimisers were built and the best (VQC-1) carried forward. No equivalent search for the classical models is described.
- **C2.2 U.** Six classical families are compared. The reported XGBoost (0.84) exceeds VQC-1 (0.83).
- **C3.1 Y.** One row per patient.
- **C3.2 N.** Ensemble feature selection was run on the full data, and the number of features was chosen from 10-fold CV performance of several classifiers (S3 notes and full text).
- **C3.3 U.**
- **C4.1 N.**
- **C4.2 N.**
- **C4.3 –.**
- **C5.1 Y.** XGBoost's higher accuracy is reported.
- **C6.1 U.** Supplementary material only.
- **C7.1 N.** The feature map is described as offering a quantum advantage because it cannot be classically simulated at scale. There is no test.
- **C7.2 –.**
- **Re-test:** quantum 85.6% vs classical 85.4% under symmetric CV. NOT SUPPORTED.

### Adaptive quantum-kernel stacking — Sci Rep 2026, PMID 42286086 (Parkinson's and diabetes claims)
- **C1.1 Y.** Neither arm is post-hoc calibrated in the headline comparison. The paper reports that the RBF baseline is better calibrated than the stack.
- **C1.2 Y.**
- **C2.1 N.** The baseline is an untuned RBF-SVM.
- **C2.2 N.** The single comparator is weak.
- **C3.1 N.** The UCI Parkinson's data have about six recordings per subject (195 recordings, 32 subjects). Stratified CV over recordings is described, with no grouping by subject.
- **C3.2 N.** The paper states that PCA was applied globally before cross-validation.
- **C3.3 Y.** Out-of-fold stacking.
- **C4.1 Y.** 1,000-resample bootstrap CIs and p-values against the baseline, plus McNemar tests.
- **C4.2 N.** Ten models are compared on several metrics with no correction stated.
- **C4.3 Y.**
- **C5.1 Y.** Weaker calibration and Breast Cancer results are reported.
- **C6.1 U.**
- **C7.1 N.** Entanglement and Hilbert-space dimension are offered as the mechanism. The paper's own ablation varies the feature maps but does not isolate entanglement.
- **C7.2 –.** Statevector.
- **Re-test:** both claims reproduce under the paper's protocol. Neither survives the symmetric protocol (Δ specificity +0.3 pts; Δ sensitivity −5.3 pts). Analysis F1/F2 separates the factors.

### Heart disease QNN vs NN — Sci Rep 2026, PMID 41513819
- **C1.1 Y.**
- **C2.1 Y.** 46 QNN configurations, and the NN is varied over hidden layers and widths.
- **C2.2 U.** Only NNs and a perceptron. No tree ensemble or logistic regression is included.
- **C2.3 N.** Up to 17,655 classical parameters against at most 250 quantum, acknowledged by the authors as intentional.
- **C3.1 Y.** One row per patient, single 70/30 split.
- **C3.2 U.**
- **C3.3 N.** Training stops when the test cost has not decreased for 500 epochs.
- **C4.1 N.** 95% CIs across runs per model. No paired test on the difference.
- **C4.2 N.**
- **C4.3 –.**
- **C5.1 Y.** Reports regimes where deeper QNNs or NNs win.
- **C6.1 U.**
- **C7.1 –.** The claim is empirical, about the low-data regime.
- **C7.2 –.** Simulation.
- **Re-test:** analysis H.

### Adolescent schizophrenia EEG QSVM — J Med Syst 2024, PMID 38441727 (read in full)
- **C1.1 Y.**
- **C2.1 N.** Thirteen feature maps and 3–15 qubits are explored for the QSVM, and the best are reported. The classical models are run at fixed settings across feature counts.
- **C2.2 U.**
- **C3.1 U.** Features come from one 1-minute recording per subject (84 subjects). Whether the classified rows are per subject or per epoch is not stated explicitly.
- **C3.2 U.** The pipeline order (features, then PCA, then classification) suggests PCA was fitted on the full dataset before the 70/30 split, but the text never says so explicitly.
- **C3.3 U.**
- **C4.1 N.** One random split, point accuracy only.
- **C4.2 N.**
- **C4.3 –.**
- **C5.1 U.** It reports degraded feature maps, but the 100% figure is presented as superior although LR also reaches 100%.
- **C6.1 U.**
- **C7.1 N.** Superposition and entanglement are offered as explanations. The two maps that reach 100% (Pauli X and Z) contain no entangling gates, as the paper itself describes them.
- **C7.2 –.** QasmSimulator.
- **Re-test:** the 100% did not reproduce on per-recording features (QSVM mean 52%). NOT SUPPORTED. Epoch-level test in F3.

### KACQ-DCNN — PMID 40865268 (screening notes only)
- **C6.1 N.** No code released (S3).
- **C3.2 U.** SMOTE in preprocessing; its timing relative to the split is not established.
- All other items **U**.
- The quantum part is a 4-qubit channel inside a 3-million-parameter KAN/BiLSTM hybrid, so a re-test could not isolate the quantum component even with code.

### PROSTATEx radiomics QSVM — SPIE 2026, PMID 42046760 (read in full)
- **C1.1 Y.**
- **C2.1 Y.** All models use the same nested stratified 5×4 CV with grid search.
- **C2.2 Y.** SVM-RBF, RF and XGBoost, all tuned.
- **C2.3 –.**
- **C3.1 Y.** Lesion-level. Patients with several lesions are possible and not discussed. Scored Y on the paper's unit.
- **C3.2 Y.** Feature selection and standardisation are fitted inside each outer fold.
- **C3.3 Y.**
- **C4.1 N.** Mean ± SD over five folds, no paired test.
- **C4.2 N.**
- **C4.3 N.**
- **C5.1 Y.** Notes XGBoost comparability and context dependence.
- **C6.1 U.**
- **C7.1 –.**
- **C7.2 –.** Statevector, stated.

### INCOV omics QSVM — PMID 41966795 (screening notes only)
Not assessable from the material retrieved. Every item is **U**.
