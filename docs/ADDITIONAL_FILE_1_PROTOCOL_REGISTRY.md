# Additional file 1 — Protocol registry

Each analysis the main text describes as pre-registered, with the repository commit that deposited its
protocol and the commit that recorded its result. Times are the commit times (UTC+05:30). A commit
timestamp is evidence of order within this repository, not an external registration; an external
deposit (OSF) of these files is pending.

| Protocol | Governs | Protocol commit | Result commit | Gap | Blind at registration |
|---|---|---|---|---:|---|
| `PREREGISTRATION_CHEXPERT.md` | §2.8 external validation (H1–H4) | `fa8c2b6` 2026-08-19 13:49 | `b792212` 2026-08-19 14:08 | 18 min | Yes |
| `PREREGISTRATION_ADDENDUM_2026-10.md` | §2.3 equivalence test (A); §2.5 patient bootstrap (B) | `3a75d19` 2026-10-02 13:36 | `d4052fb` 2026-10-02 14:01 | 25 min | No: point estimates already published |
| `PREREGISTRATION_ADDENDUM_2026-10.md` | §2.5 re-splits (C) and re-uploading ablation (D) | `3a75d19` 2026-10-02 13:36 | `d4052fb` 2026-10-02 14:01 | 25 min | Yes |
| `PREREGISTRATION_ADDENDUM_2026-10b_CHEXPERT_INTERNAL.md` | §2.8 out-of-sample matched-optimiser comparison (E1); §2.5 out-of-sample ablation (E2) | `3a54488` 2026-10-02 21:06 | `76eacb7` 2026-10-02 22:03 | 57 min | Yes |

**Corroborating output times.** File modification times of the outputs (weaker evidence than commits, since they can be altered) agree with the order above: addendum (a) analysis A wrote `A_tost.json` at 13:36:43, eleven seconds after its protocol commit, and analyses C, B and D finished at 13:45:57, 13:53:52 and 13:59:37; addendum (b) analyses E1 and E2 finished at 21:41:58 and 22:00:09.

**Pre-specified but not committed before their results.** These protocols state in their text that
they were written before any model was run, but neither was committed before its results existed, so
their timing cannot be verified from the repository. The main text calls them *pre-specified*, not
pre-registered.

| Protocol | Governs | Status |
|---|---|---|
| `FIELD_AUDIT_PROTOCOL.md` | §2.1 field audit | Dated 21 September 2026 in its text; not committed |
| `PREREGISTRATION_QKL_BRATS2020.md` | §2.7 glioma kernel re-analysis | Dated 21 September 2026 in its text; not committed. Names the 369-subject analysis primary; the main text reports the 111-subject sensitivity analysis first, as a stated deviation |

**Order of the revision analyses.** The equivalence tests, patient bootstrap, re-splits, re-uploading
ablation and out-of-sample replication were added after earlier results were known, in response to
critique of earlier manuscript versions. Their protocols precede their runs, as the table shows; the
decision to run them did not precede the earlier results.
