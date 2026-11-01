# Letters to the authors of the audited studies — DRAFTS, NOT SENT

Purpose: before submission, give each group our findings and code, ask for corrections, and
report their replies (or non-replies) in the manuscript. These are drafts for the author to
review, address and send personally. Corresponding-author addresses are taken from each paper's
own correspondence line; none are filled in here. Record the date sent and any reply in
`docs/correspondence/REPLIES.md`.

Numbers marked ⟨F1⟩/⟨F2⟩/⟨F3⟩/⟨H⟩ are filled from the pre-registered v6 analyses once they finish
(`bench/field_audit/results/factors_*.json`, `heart_qnn.json`); do not send before they are.

---

## Common text (adapt per letter)

Subject: Re-analysis of your [journal, year] study — findings and code shared before publication

Dear Dr [surname],

I am writing about your article "[title]" ([journal] [year]; doi:[doi]). As part of a
methods study on how claims of benefit from novel model components in clinical machine learning
hold up under symmetric evaluation, I re-implemented your pipeline from the paper and re-tested
its central claim. I am contacting you before submission so that you can point out anything I
have misread or implemented differently from your work, and so that your response can be
reported alongside the results.

Our protocol, written before any study was run, has two arms: a reproduction arm following your
described protocol as closely as the paper allows (choices where the paper is silent are
recorded), and a symmetric arm in which the quantum model and a nested-selected classical
comparator (logistic regression, RBF-SVM, random forest, gradient boosting, MLP; 16
configurations) are evaluated on identical repeated cross-validation folds with
leakage-free preprocessing and the Nadeau–Bengio corrected resampled t-test.

[STUDY-SPECIFIC PARAGRAPH]

The code, results files and protocol are public: github.com/Sivaguru-2008/aura
(`bench/field_audit/`, `docs/FIELD_AUDIT_PROTOCOL.md`). If your own code or exact splits are
available, I would gladly re-run with them and report any difference. I would be grateful for
any correction, and with your permission I will report your reply in the paper; if you prefer
not to reply, the paper will say only that the authors were contacted on [date].

With thanks and kind regards,
Sivaguru R. M.
Independent researcher · sivagurumurugan1@gmail.com

---

## 1. CKD — Parthasarathi et al., PeerJ Comput Sci 2025;11:e2789 (PMID 40567645)

Under our reading of your protocol, the VQC reached 83.5% accuracy rather than the reported
99.2%, while SVM, random forest and MLP reached 96.0–97.8%, above the figures reported for them.
Under the symmetric protocol (25 outer folds) the VQC reached 87.7% against 97.4% for the
nested-selected classical arm (Δ −9.65 points, 95% CI −14.9 to −4.4, p = 0.0009). The points
where our implementation may differ from yours are the sparse-autoencoder augmentation (whether
synthetic rows were generated before or after the train/test split), the LASSO selection step,
and the amplitude-encoding normalisation. Could you tell me how these were done, and whether
the 99.2% was measured on a held-out set that contained no augmented rows?

## 2. Glioma grading — Akpinar & Oduncuoglu, Sci Rep 2025;15:25935 (PMID 40676161)

We reproduced the classical models at or above the reported figures (SVC 85.3% against the
reported 75%). Under the symmetric protocol, VQC accuracy was 85.6% against 85.4% for the
nested-selected classical arm (Δ +0.19 points, 95% CI −1.2 to +1.6, p = 0.78), so we could not
confirm an advantage. Two choices may matter: the ensemble feature selection appears to have
been run on the full dataset before cross-validation, and the number of features was chosen
from cross-validated classifier performance. Was feature selection repeated inside each fold in
your 5-fold analysis, and were the classical models tuned?

## 3. Quantum-kernel stacking — Daga et al., Sci Rep 2026;16:26840 (PMID 42286086)

Your claims reproduce under your protocol: stack specificity 87.5% against 60.4% on Parkinson's,
and sensitivity 69.8% against 56.7% on Pima. Under the symmetric protocol they do not
(Parkinson's specificity 43.3% vs 43.0%; Pima sensitivity 69.3% vs 74.6%). A pre-registered
factorial analysis then separated three differences between the protocols: PCA fitted globally
or inside folds, cross-validation by recording or grouped by subject (the UCI Parkinson's data
contain about six recordings per subject), and an untuned versus a nested-selected comparator.
⟨F1: which single change removes the advantage, with Δ and CI⟩. ⟨F2: same for Pima⟩. We note
your own report that per-fold PCA left Parkinson's accuracy unchanged, which is consistent with
⟨F1 PCA main effect⟩. Was subject grouping used anywhere in your cross-validation?

## 4. Schizophrenia EEG — Aksoy et al., J Med Syst 2024;48:29 (PMID 38441727)

On per-recording features (one row per subject; DWT statistics on Cz), across 200 random 70/30
splits the QSVM averaged 52% and logistic regression 69%, and no split reached 100%. We then
tested a hypothesis about how 100% could arise: ⟨F3: with epochs from the same subject on both
sides of a random split, mean accuracy X; grouped by subject, Y⟩. We do not know whether your
pipeline split at the recording or the epoch level. Could you tell me the unit of each row in
your classification table, whether PCA was fitted before the split, and the random seed?

## 5. Heart disease QNN — PMID 41513819, Sci Rep 2026

⟨H: symmetric-arm results at 10/30/50% training fractions⟩. In the paper, training stops when
the test-set cost stops decreasing; in our reproduction arm we emulated that, and in the
symmetric arm stopping used a validation split carved from the training data. Is that reading
of your stopping rule correct?

---

Not written: KACQ-DCNN (PMID 40865268) — no code was released, so there is nothing to re-test
against; a short request for code could be sent. PROSTATEx (PMID 42046760) and INCOV
(PMID 41966795) — eligible but not run.
