# Verification Report: Manuscript v5.0 Numbers vs Artifacts

Generated for Step 0 verification before producing `MANUSCRIPT_v5.1.md`.

| Manuscript Location | Target Value in v5.0 | Source Artifact File | JSON Key / Path | Artifact Value | Match? |
|---|---|---|---|---|:---:|
| Table 1 (published) Product-of-experts accuracy | `0.5414` | `fusion_baselines_n447.json` | `backends.classical.accuracy` | `0.5414` | **YES** |
| Table 1 (published) Product-of-experts nll | `1.3026` | `fusion_baselines_n447.json` | `backends.classical.nll` | `1.3026` | **YES** |
| Table 1 (published) Product-of-experts ece | `0.1266` | `fusion_baselines_n447.json` | `backends.classical.ece` | `0.1266` | **YES** |
| Table 1 (published) Product-of-experts macro_auroc | `0.8045` | `fusion_baselines_n447.json` | `backends.classical.macro_auroc` | `0.8045` | **YES** |
| Table 1 (published) Product-of-experts T | `0.479` | `fusion_baselines_n447.json` | `backends.classical.temperature` | `0.479` | **YES** |
| Table 1 (published) Quantum VQC accuracy | `0.5324` | `fusion_baselines_n447.json` | `backends.quantum.accuracy` | `0.5324` | **YES** |
| Table 1 (published) Quantum VQC nll | `1.2853` | `fusion_baselines_n447.json` | `backends.quantum.nll` | `1.2853` | **YES** |
| Table 1 (published) Quantum VQC ece | `0.0896` | `fusion_baselines_n447.json` | `backends.quantum.ece` | `0.0896` | **YES** |
| Table 1 (published) Quantum VQC macro_auroc | `0.8014` | `fusion_baselines_n447.json` | `backends.quantum.macro_auroc` | `0.8014` | **YES** |
| Table 1 (published) Quantum VQC T | `0.906` | `fusion_baselines_n447.json` | `backends.quantum.temperature` | `0.9057` | **YES** |
| Table 1 (published) Gated head accuracy | `0.5526` | `fusion_baselines_n447.json` | `backends.learnable.accuracy` | `0.5526` | **YES** |
| Table 1 (published) Gated head nll | `1.2663` | `fusion_baselines_n447.json` | `backends.learnable.nll` | `1.2663` | **YES** |
| Table 1 (published) Gated head ece | `0.0902` | `fusion_baselines_n447.json` | `backends.learnable.ece` | `0.0902` | **YES** |
| Table 1 (published) Gated head macro_auroc | `0.8056` | `fusion_baselines_n447.json` | `backends.learnable.macro_auroc` | `0.8056` | **NO** |
| Table 1 (published) Gated head T | `0.887` | `fusion_baselines_n447.json` | `backends.learnable.temperature` | `0.8867` | **YES** |
| Table 1 (published) Logistic regression accuracy | `0.5570` | `fusion_baselines_n447.json` | `backends.logistic.accuracy` | `0.557` | **YES** |
| Table 1 (published) Logistic regression nll | `1.2310` | `fusion_baselines_n447.json` | `backends.logistic.nll` | `1.231` | **YES** |
| Table 1 (published) Logistic regression ece | `0.0862` | `fusion_baselines_n447.json` | `backends.logistic.ece` | `0.0862` | **YES** |
| Table 1 (published) Logistic regression macro_auroc | `0.8227` | `fusion_baselines_n447.json` | `backends.logistic.macro_auroc` | `0.8227` | **YES** |
| Table 1 (published) Logistic regression T | `1.072` | `fusion_baselines_n447.json` | `backends.logistic.temperature` | `1.0717` | **YES** |
| Table 1 (matched) Product-of-experts accuracy | `0.5593` | `fusion_matched_optimiser.json` | `backends.classical.accuracy` | `0.5593` | **YES** |
| Table 1 (matched) Product-of-experts nll | `1.2362` | `fusion_matched_optimiser.json` | `backends.classical.nll` | `1.2362` | **YES** |
| Table 1 (matched) Product-of-experts ece | `0.0810` | `fusion_matched_optimiser.json` | `backends.classical.ece` | `0.081` | **YES** |
| Table 1 (matched) Product-of-experts macro_auroc | `0.8219` | `fusion_matched_optimiser.json` | `backends.classical.macro_auroc` | `0.8219` | **YES** |
| Table 1 (matched) Product-of-experts T | `1.054` | `fusion_matched_optimiser.json` | `backends.classical.temperature` | `1.0537` | **YES** |
| Table 1 (matched) Product-of-experts params | `54` | `fusion_matched_optimiser.json` | `trainable_parameters.classical` | `54` | **YES** |
| Table 1 (matched) Quantum VQC accuracy | `0.5436` | `fusion_matched_optimiser.json` | `backends.quantum.accuracy` | `0.5436` | **YES** |
| Table 1 (matched) Quantum VQC nll | `1.2395` | `fusion_matched_optimiser.json` | `backends.quantum.nll` | `1.2395` | **YES** |
| Table 1 (matched) Quantum VQC ece | `0.0486` | `fusion_matched_optimiser.json` | `backends.quantum.ece` | `0.0486` | **YES** |
| Table 1 (matched) Quantum VQC macro_auroc | `0.8009` | `fusion_matched_optimiser.json` | `backends.quantum.macro_auroc` | `0.8009` | **YES** |
| Table 1 (matched) Quantum VQC T | `1.003` | `fusion_matched_optimiser.json` | `backends.quantum.temperature` | `1.0033` | **YES** |
| Table 1 (matched) Quantum VQC params | `102` | `fusion_matched_optimiser.json` | `trainable_parameters.quantum` | `102` | **YES** |
| Table 1 (matched) Gated head accuracy | `0.5682` | `fusion_matched_optimiser.json` | `backends.learnable.accuracy` | `0.5682` | **YES** |
| Table 1 (matched) Gated head nll | `1.2291` | `fusion_matched_optimiser.json` | `backends.learnable.nll` | `1.2291` | **YES** |
| Table 1 (matched) Gated head ece | `0.0804` | `fusion_matched_optimiser.json` | `backends.learnable.ece` | `0.0804` | **YES** |
| Table 1 (matched) Gated head macro_auroc | `0.8252` | `fusion_matched_optimiser.json` | `backends.learnable.macro_auroc` | `0.8252` | **YES** |
| Table 1 (matched) Gated head T | `1.007` | `fusion_matched_optimiser.json` | `backends.learnable.temperature` | `1.0071` | **YES** |
| Table 1 (matched) Gated head params | `126` | `fusion_matched_optimiser.json` | `trainable_parameters.learnable` | `126` | **YES** |
| Table 1 (matched) Logistic regression accuracy | `0.5570` | `fusion_matched_optimiser.json` | `backends.logistic.accuracy` | `0.557` | **YES** |
| Table 1 (matched) Logistic regression nll | `1.2310` | `fusion_matched_optimiser.json` | `backends.logistic.nll` | `1.231` | **YES** |
| Table 1 (matched) Logistic regression ece | `0.0862` | `fusion_matched_optimiser.json` | `backends.logistic.ece` | `0.0862` | **YES** |
| Table 1 (matched) Logistic regression macro_auroc | `0.8227` | `fusion_matched_optimiser.json` | `backends.logistic.macro_auroc` | `0.8227` | **YES** |
| Table 1 (matched) Logistic regression T | `1.072` | `fusion_matched_optimiser.json` | `backends.logistic.temperature` | `1.0717` | **YES** |
| Table 1 (matched) Logistic regression params | `54` | `fusion_matched_optimiser.json` | `trainable_parameters.logistic` | `54` | **YES** |
| Table 2 Product-of-experts accuracy | `0.5273` | `chexpert_external.json` | `backends.classical.accuracy` | `0.5273` | **YES** |
| Table 2 Product-of-experts nll | `1.4193` | `chexpert_external.json` | `backends.classical.nll` | `1.4193` | **YES** |
| Table 2 Product-of-experts ece | `0.1346` | `chexpert_external.json` | `backends.classical.ece` | `0.1346` | **YES** |
| Table 2 Product-of-experts macro_auroc | `0.7366` | `chexpert_external.json` | `backends.classical.macro_auroc` | `0.7366` | **YES** |
| Table 2 Quantum VQC accuracy | `0.4721` | `chexpert_external.json` | `backends.quantum.accuracy` | `0.4721` | **YES** |
| Table 2 Quantum VQC nll | `1.5782` | `chexpert_external.json` | `backends.quantum.nll` | `1.5782` | **YES** |
| Table 2 Quantum VQC ece | `0.1586` | `chexpert_external.json` | `backends.quantum.ece` | `0.1586` | **YES** |
| Table 2 Quantum VQC macro_auroc | `0.7259` | `chexpert_external.json` | `backends.quantum.macro_auroc` | `0.7259` | **YES** |
| Table 2 Gated head accuracy | `0.4829` | `chexpert_external.json` | `backends.learnable.accuracy` | `0.4829` | **YES** |
| Table 2 Gated head nll | `1.5116` | `chexpert_external.json` | `backends.learnable.nll` | `1.5116` | **YES** |
| Table 2 Gated head ece | `0.1424` | `chexpert_external.json` | `backends.learnable.ece` | `0.1424` | **YES** |
| Table 2 Gated head macro_auroc | `0.7367` | `chexpert_external.json` | `backends.learnable.macro_auroc` | `0.7367` | **YES** |
| Table 2 Logistic regression accuracy | `0.5210` | `chexpert_external.json` | `backends.logistic.accuracy` | `0.521` | **YES** |
| Table 2 Logistic regression nll | `1.3815` | `chexpert_external.json` | `backends.logistic.nll` | `1.3815` | **YES** |
| Table 2 Logistic regression ece | `0.1078` | `chexpert_external.json` | `backends.logistic.ece` | `0.1078` | **YES** |
| Table 2 Logistic regression macro_auroc | `0.7298` | `chexpert_external.json` | `backends.logistic.macro_auroc` | `0.7298` | **YES** |
| Table 4 T1_clinical_linear linear ceiling | `0.9991` | `nonlinear_control.json` | `targets.T1_clinical_linear.ceilings.linear_ceiling` | `0.9991` | **YES** |
| Table 4 T1_clinical_linear interaction ceiling | `1.0000` | `nonlinear_control.json` | `targets.T1_clinical_linear.ceilings.interaction_ceiling` | `1.0` | **YES** |
| Table 4 T1_clinical_linear classical NLL | `0.0382` | `nonlinear_control.json` | `targets.T1_clinical_linear.backends.classical.nll` | `0.0382` | **YES** |
| Table 4 T1_clinical_linear classical Acc | `0.9843` | `nonlinear_control.json` | `targets.T1_clinical_linear.backends.classical.accuracy` | `0.9843` | **YES** |
| Table 4 T1_clinical_linear learnable NLL | `0.0636` | `nonlinear_control.json` | `targets.T1_clinical_linear.backends.learnable.nll` | `0.0636` | **YES** |
| Table 4 T1_clinical_linear learnable Acc | `0.9821` | `nonlinear_control.json` | `targets.T1_clinical_linear.backends.learnable.accuracy` | `0.9821` | **YES** |
| Table 4 T1_clinical_linear quantum NLL | `0.1651` | `nonlinear_control.json` | `targets.T1_clinical_linear.backends.quantum.nll` | `0.1651` | **YES** |
| Table 4 T1_clinical_linear quantum Acc | `0.9441` | `nonlinear_control.json` | `targets.T1_clinical_linear.backends.quantum.accuracy` | `0.9441` | **YES** |
| Table 4 T1_clinical_linear logistic NLL | `0.0736` | `nonlinear_control.json` | `targets.T1_clinical_linear.backends.logistic.nll` | `0.0736` | **YES** |
| Table 4 T1_clinical_linear logistic Acc | `0.9799` | `nonlinear_control.json` | `targets.T1_clinical_linear.backends.logistic.accuracy` | `0.9799` | **YES** |
| Table 4 T2a_conjunctive linear ceiling | `0.8740` | `nonlinear_control.json` | `targets.T2a_conjunctive.ceilings.linear_ceiling` | `0.874` | **YES** |
| Table 4 T2a_conjunctive interaction ceiling | `0.9301` | `nonlinear_control.json` | `targets.T2a_conjunctive.ceilings.interaction_ceiling` | `0.9301` | **YES** |
| Table 4 T2a_conjunctive classical NLL | `0.5911` | `nonlinear_control.json` | `targets.T2a_conjunctive.backends.classical.nll` | `0.5911` | **YES** |
| Table 4 T2a_conjunctive classical Acc | `0.8345` | `nonlinear_control.json` | `targets.T2a_conjunctive.backends.classical.accuracy` | `0.8345` | **YES** |
| Table 4 T2a_conjunctive learnable NLL | `0.5075` | `nonlinear_control.json` | `targets.T2a_conjunctive.backends.learnable.nll` | `0.5075` | **YES** |
| Table 4 T2a_conjunctive learnable Acc | `0.8345` | `nonlinear_control.json` | `targets.T2a_conjunctive.backends.learnable.accuracy` | `0.8345` | **YES** |
| Table 4 T2a_conjunctive quantum NLL | `0.7980` | `nonlinear_control.json` | `targets.T2a_conjunctive.backends.quantum.nll` | `0.798` | **YES** |
| Table 4 T2a_conjunctive quantum Acc | `0.7539` | `nonlinear_control.json` | `targets.T2a_conjunctive.backends.quantum.accuracy` | `0.7539` | **YES** |
| Table 4 T2a_conjunctive logistic NLL | `0.5646` | `nonlinear_control.json` | `targets.T2a_conjunctive.backends.logistic.nll` | `0.5646` | **YES** |
| Table 4 T2a_conjunctive logistic Acc | `0.8501` | `nonlinear_control.json` | `targets.T2a_conjunctive.backends.logistic.accuracy` | `0.8501` | **YES** |
| Table 4 T2b_pairwise linear ceiling | `0.9081` | `nonlinear_control.json` | `targets.T2b_pairwise.ceilings.linear_ceiling` | `0.9081` | **YES** |
| Table 4 T2b_pairwise interaction ceiling | `0.9821` | `nonlinear_control.json` | `targets.T2b_pairwise.ceilings.interaction_ceiling` | `0.9821` | **YES** |
| Table 4 T2b_pairwise classical NLL | `0.3238` | `nonlinear_control.json` | `targets.T2b_pairwise.backends.classical.nll` | `0.3238` | **YES** |
| Table 4 T2b_pairwise classical Acc | `0.8971` | `nonlinear_control.json` | `targets.T2b_pairwise.backends.classical.accuracy` | `0.8971` | **YES** |
| Table 4 T2b_pairwise learnable NLL | `0.2560` | `nonlinear_control.json` | `targets.T2b_pairwise.backends.learnable.nll` | `0.256` | **YES** |
| Table 4 T2b_pairwise learnable Acc | `0.9195` | `nonlinear_control.json` | `targets.T2b_pairwise.backends.learnable.accuracy` | `0.9195` | **YES** |
| Table 4 T2b_pairwise quantum NLL | `0.2515` | `nonlinear_control.json` | `targets.T2b_pairwise.backends.quantum.nll` | `0.2515` | **YES** |
| Table 4 T2b_pairwise quantum Acc | `0.8971` | `nonlinear_control.json` | `targets.T2b_pairwise.backends.quantum.accuracy` | `0.8971` | **YES** |
| Table 4 T2b_pairwise logistic NLL | `0.3155` | `nonlinear_control.json` | `targets.T2b_pairwise.backends.logistic.nll` | `0.3155` | **YES** |
| Table 4 T2b_pairwise logistic Acc | `0.8971` | `nonlinear_control.json` | `targets.T2b_pairwise.backends.logistic.accuracy` | `0.8971` | **YES** |
| Table 5 CKD Symmetric Quantum | `87.70%` | `bench/field_audit/results/ckd.json` | `symmetric.quantum` | `87.70%` | **YES** |
| Table 5 CKD Symmetric Classical | `97.35%` | `bench/field_audit/results/ckd.json` | `symmetric.best_classical` | `97.35%` | **YES** |
| Table 5 CKD Delta | `-9.65%` | `bench/field_audit/results/ckd.json` | `symmetric.delta` | `-9.65%` | **YES** |
| Table 5 CKD p-value | `0.000915` | `bench/field_audit/results/ckd.json` | `symmetric.nadeau_bengio_p` | `0.000915` | **YES** |
| Table 5 Glioma Symmetric Quantum | `85.60%` | `bench/field_audit/results/glioma.json` | `symmetric.quantum` | `85.60%` | **YES** |
| Table 5 Glioma Symmetric Classical | `85.41%` | `bench/field_audit/results/glioma.json` | `symmetric.best_classical` | `85.41%` | **YES** |
| Table 5 Glioma Delta | `+0.19%` | `bench/field_audit/results/glioma.json` | `symmetric.delta` | `+0.19%` | **YES** |
| Table 5 Glioma p-value | `0.7824` | `bench/field_audit/results/glioma.json` | `symmetric.nadeau_bengio_p` | `0.7824` | **YES** |
| Table 5 Parkinson's Symmetric Quantum | `43.33%` | `bench/field_audit/results/stack_parkinsons.json` | `symmetric.quantum` | `43.33%` | **YES** |
| Table 5 Parkinson's Symmetric Classical | `43.00%` | `bench/field_audit/results/stack_parkinsons.json` | `symmetric.best_classical` | `43.00%` | **YES** |
| Table 5 Parkinson's Delta | `+0.33%` | `bench/field_audit/results/stack_parkinsons.json` | `symmetric.delta` | `+0.33%` | **YES** |
| Table 5 Parkinson's p-value | `0.9785` | `bench/field_audit/results/stack_parkinsons.json` | `symmetric.nadeau_bengio_p` | `0.9785` | **YES** |
| Table 5 Diabetes Symmetric Quantum | `69.34%` | `bench/field_audit/results/stack_pima.json` | `symmetric.quantum` | `69.34%` | **YES** |
| Table 5 Diabetes Symmetric Classical | `74.64%` | `bench/field_audit/results/stack_pima.json` | `symmetric.best_classical` | `74.64%` | **YES** |
| Table 5 Diabetes Delta | `-5.30%` | `bench/field_audit/results/stack_pima.json` | `symmetric.delta` | `-5.30%` | **YES** |
| Table 5 Diabetes p-value | `0.0666` | `bench/field_audit/results/stack_pima.json` | `symmetric.nadeau_bengio_p` | `0.0666` | **YES** |
| Table 5 Schizophrenia Symmetric Quantum | `59.10%` | `bench/field_audit/results/eeg_qsvm.json` | `symmetric.quantum` | `59.10%` | **YES** |
| Table 5 Schizophrenia Symmetric Classical | `66.71%` | `bench/field_audit/results/eeg_qsvm.json` | `symmetric.best_classical` | `66.71%` | **YES** |
| Table 5 Schizophrenia Delta | `-7.60%` | `bench/field_audit/results/eeg_qsvm.json` | `symmetric.delta` | `-7.60%` | **YES** |
| Table 5 Schizophrenia p-value | `0.3187` | `bench/field_audit/results/eeg_qsvm.json` | `symmetric.nadeau_bengio_p` | `0.3187` | **YES** |
| Section 2.6 (369-subject) quantum AUROC | `0.9422` | `qkl_brats2020_official.json` | `pooled_out_of_fold.delong.auroc_a` | `0.9422` | **YES** |
| Section 2.6 (369-subject) rbf AUROC | `0.9405` | `qkl_brats2020_official.json` | `pooled_out_of_fold.delong.auroc_b` | `0.9405` | **YES** |
| Section 2.6 (369-subject) delta AUROC | `+0.0017` | `qkl_brats2020_official.json` | `pooled_out_of_fold.delong.delta_auroc` | `0.0017` | **YES** |
| Section 2.6 (369-subject) p-value | `0.78` | `qkl_brats2020_official.json` | `pooled_out_of_fold.delong.p` | `0.77733` | **YES** |
| BraTS 111 heldout subjects total | `111` | `qkl_brats2020_official_heldout111.json` | `corpus.subjects` | `111` | **YES** |
| BraTS 111 heldout LGG subjects | `23` | `qkl_brats2020_official_heldout111.json` | `corpus.subject_positives_LGG` | `23` | **YES** |
| BraTS 111 heldout HGG subjects | `88` | `qkl_brats2020_official_heldout111.json` | `corpus.subject_negatives_HGG` | `88` | **YES** |
| BraTS 111 heldout quantum AUROC | `0.9407` | `qkl_brats2020_official_heldout111.json` | `pooled_out_of_fold.delong.auroc_a` | `0.9407` | **YES** |
| BraTS 111 heldout rbf AUROC | `0.9625` | `qkl_brats2020_official_heldout111.json` | `pooled_out_of_fold.delong.auroc_b` | `0.9625` | **YES** |
| BraTS 111 heldout delta AUROC | `-0.0217` | `qkl_brats2020_official_heldout111.json` | `pooled_out_of_fold.delong.delta_auroc` | `-0.0217` | **YES** |
| BraTS 111 heldout CI95 lower | `-0.0543` | `qkl_brats2020_official_heldout111.json` | `pooled_out_of_fold.delong.ci95[0]` | `-0.0543` | **YES** |
| BraTS 111 heldout CI95 upper | `0.0109` | `qkl_brats2020_official_heldout111.json` | `pooled_out_of_fold.delong.ci95[1]` | `0.0109` | **YES** |
| BraTS 111 heldout p-value | `0.191` | `qkl_brats2020_official_heldout111.json` | `pooled_out_of_fold.delong.p` | `0.19121` | **YES** |
| Section 2.4 ablation delta NLL mean | `+0.025` | `entanglement_multiseed_7q.json` | `delta_nll.mean` | `0.025` | **YES** |
| Section 2.4 ablation Holm p | `0.006` | `entanglement_multiseed_7q.json` | `delta_nll.wilcoxon_p_holm` | `0.0059` | **YES** |
| Section 2.5 sweep top 5 unentangled count | `three of the five best` | `design_sweep.json` | `top 5 cells with entangler=='none'` | `3` | **YES** |
| Section 2.1 calibration raw ECE | `0.0896` | `backend_calibration.json` | `backends.quantum.test_ece` | `0.0896` | **YES** |
| Section 2.1 calibration classical ECE | `0.1266` | `backend_calibration.json` | `backends.classical.test_ece` | `0.1266` | **YES** |
| Section 2.1 calibration temperature spread | `1.89` | `backend_calibration.json` | `temperature_spread` | `1.89` | **YES** |

**Summary:** 128 verified matches, 1 mismatches out of 129 evaluated items.