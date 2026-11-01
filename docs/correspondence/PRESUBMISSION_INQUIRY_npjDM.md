# Pre-submission inquiry — npj Digital Medicine (DRAFT, not sent)

**Before sending, verify on the journal site (login-gated; could not be fetched):**

1. That pre-submission enquiries are accepted, and through which route (online system or email).
2. The current article-processing charge and any waiver policy.
3. The Article limits: abstract ≤150 words unstructured, title ≤15 words, main text about
   3,000–4,000 words. These match the format guidance found by search but are unconfirmed.

Send only after the confirmatory result (⟨C1⟩) is in, so that the pitch states it.

---

Subject: Pre-submission enquiry — Do claimed benefits of a novel component in clinical AI survive symmetric evaluation? A pre-registered audit using quantum machine learning as the case study

Dear Editors,

I would be grateful for your view on whether the following manuscript is within scope for
*npj Digital Medicine* before I submit it.

**The question.** Clinical AI papers often report that a new model component improves
prediction. Such a benefit can come from the component, or from evaluating the component more
carefully than its comparator: more tuning, calibration applied to one arm only, a weaker
baseline, preprocessing fitted before the split. Reviewers rarely see these choices. We measured
how much of a reported benefit survives when every arm is evaluated the same way. Quantum
machine learning is our case study, because your journal's recent systematic review (Gupta et
al., 2025) found its claims numerous, its comparators often ad hoc, and its differences
negligible on synthesis.

**What we did.**

1. In a clinical decision-support system we built for chest radiography, with a quantum fusion
   layer we served and promoted ourselves, we showed that the measured benefit changed sign with
   three choices: calibration symmetry, choice of comparator, and optimiser. None of our own six
   claims of quantum benefit survived, and one was retracted after reading the code.
2. We re-tested published claims of quantum benefit on clinical prediction tasks. None was
   supported under symmetric evaluation, and one reversed significantly. A pre-registered
   factorial analysis identified which evaluation choice produced each published advantage.
3. We scored all eight eligible studies against a 17-item symmetric-evaluation checklist,
   committed before scoring. Five of six assessable studies were at risk of asymmetric
   evaluation.
4. In a pre-registered confirmatory test on ⟨n⟩ chest radiographs that no model or analysis had
   seen, ⟨C1 one-sentence result⟩.
5. We explain the null mechanistically: the circuit computes a low-order trigonometric
   function, and its encoding kernel equals a classical closed form. A positive control shows
   that the same comparison detects a quantum benefit when the target contains one.

**Why *npj Digital Medicine*.** The contribution is a validated evaluation method, the checklist
and protocol, for deciding whether a claimed improvement in a clinical AI model is real. It is
relevant to reviewers, procurement and regulators well beyond quantum methods. Code, every
pre-registration with its commit timestamp, and a one-command reproduction are public.

**Limitations we state plainly.** The chest-radiograph target is derived from reports by a
fixed rule, which is what makes the positive control possible, and it is not a clinician's
diagnosis. The quantum results concern 7–10 qubits under exact simulation. The audited studies
were selected by feasibility. I am currently the sole author and am seeking a clinical
co-author.

Abstract attached. I would be glad to know whether the editors would consider a full submission.

Yours sincerely,
Sivaguru R. M.
Independent researcher · sivagurumurugan1@gmail.com
