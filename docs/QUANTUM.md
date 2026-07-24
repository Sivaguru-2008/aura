# AURA — the quantum layer

What the quantum part does, what it measurably buys, and what it measurably does not.
Every number here is reproducible from `artifacts/quantum_study.json`, measured on
447 held-out, patient-disjoint real MIMIC-CXR studies, with each model scored at its
own fitted temperature.

```bash
python -m ml.evaluation.quantum_study --n 3000     # regenerates every number below
python -m ml.training.recalibrate_backend          # per-backend calibration
python -m pytest tests/test_quantum_measurement.py # 30 tests
```

---

## 1. Where quantum sits

```
X-ray → DenseNet-121 vision → 8 evidence channels → [ QUANTUM FUSION ] → safety → report
                                                            ↑
                                                    the only quantum stage
```

Quantum is not doing the seeing — a CNN does that. Quantum does the **deciding**. The
eight channels (opacity, consolidation, effusion, cardiomegaly, nodule,
hyperinflation, pneumothorax, prior-risk) are angle-encoded onto eight qubits,
entangled through a CNOT ring, and read out as `<Z_i>` into a diagnosis head.

Eight qubits is not a limitation to apologise for — it is the design. Compressing to
clinically meaningful evidence dimensions is what keeps `n` small, and small `n` is
the only lever that attacks the barren-plateau exponent `Var[∂C] ~ 2^-n` directly.

---

## 2. The four questions, answered with numbers

### Q1 — Does entanglement do clinical work?

Ablation: the shipped CNOT-ring VQC against an otherwise-identical product-state VQC
(`entangler="none"`). Same qubits, same layers, **same 102 trainable parameters**,
same encoding, same readout, same optimiser, same seed, same data. The two-qubit
gates are the only difference, so any delta is attributable to entanglement and to
nothing else.

| | accuracy | NLL | ECE |
|---|---|---|---|
| VQC entangled (ring) | 0.5414 | 1.2682 | **0.0464** |
| VQC product (no CNOTs) | **0.5705** | **1.2120** | 0.0831 |
| Classical PoE | 0.5145 | 1.3230 | 0.0834 |

Paired bootstrap, 2000 resamples, n=447:

| Δ (entangled − product) | estimate | 95% CI | significant |
|---|---|---|---|
| accuracy | −0.0288 | [−0.0582, +0.0000] | no |
| **NLL** | **+0.0560** | **[+0.0222, +0.0902]** | **yes** |
| ECE | −0.0273 | [−0.0676, +0.0180] | no |

**The honest answer: entanglement does not help, and on log-likelihood it
significantly hurts.** The product-state VQC is the better model on this task.

This is a negative result and it is reported as one. It is also the single most
useful thing in this document, because it is the question every quantum judge asks
and almost nobody answers with a controlled experiment.

Two caveats that cut in the other direction and are stated for completeness rather
than rescue: the entangled VQC has the **best calibration of all three models**
(ECE 0.046 against 0.083 for both alternatives), and the accuracy CI touches zero at
its upper bound. Neither reaches significance for the entangled-vs-product contrast.

### Q2 — Is quantum competitive with classical?

Yes, on this split, and by more than the ablation suggests: **both** VQC variants beat
the classical product-of-experts on accuracy and NLL, and the entangled VQC halves its
calibration error.

This supersedes the earlier `registry.json` numbers, which were measured on a
173-study split. Larger split, same protocol, different conclusion — reported here
rather than quietly replaced.

### Q3 — What does a measurement budget buy? *(the part with no classical analogue)*

On hardware you do not get an expectation value, you get an estimate, and its
precision is bought with measurements: `Var[<Z_i>] = (1 − <Z_i>²)/n_shots`.

**Quantum Measurement-Budgeted Abstention** (`services/fusion/qmba.py`) turns that
into a sequential decision procedure. Start at 128 shots. Estimate the decision margin
`p_top1 − p_top2` and its shot-noise spread. Commit if separated; otherwise double the
budget; abstain at the ceiling.

Measured over 447 studies:

| | |
|---|---|
| Commit rate | 86.4% |
| Accuracy at commit | **0.5725** (vs 0.5414 overall — abstention improves what remains) |
| **Median shots spent** | **128** — the minimum probe |
| Mean shots spent | 916 |
| vs a fixed 8192-shot budget | **64× fewer measurements at the median, 8.9× at the mean** |

And the part that matters clinically — abstentions are *typed*:

| Abstention type | n | What to do |
|---|---|---|
| Model-limited | 60 | Escalate. No achievable budget separates these two diagnoses. |
| Measurement-limited | 1 | Run the circuit longer. ~8,433 shots would resolve it. |

Two different instructions from one system. A classical model cannot make this
distinction, because classical inference has no measurement budget to vary.

The 60:1 ratio is itself a finding: on this system the bottleneck is the **model**,
not the quantum hardware.

### Q4 — What does the entanglement encode?

`C_ij = <Z_i Z_j> − <Z_i><Z_j>`, measured in the same pass as the prediction. Zero on
a product state (`tests/test_quantum_measurement.py` asserts this to 1e-12), non-zero
only when the CNOT ring has coupled the channels. It is a property of the state that
produced the diagnosis, not an attribution obtained by perturbing inputs.

Top consistently-coupled evidence pairs across the test set:

| pair | mean abs differential |
|---|---|
| opacity ↔ nodule | 0.517 |
| consolidation ↔ cardiomegaly | 0.418 |
| effusion ↔ nodule | 0.377 |

**One correction that mattered.** The raw correlator is *not* attributable to the
patient: the trained rotations entangle the register regardless of input, so an empty
evidence vector measures total coupling 6.19 while a realistic effusion-plus-
cardiomegaly vector measures 2.29 — the emptiest study looks the most coupled.
Reporting the raw number would have been backwards. The reportable quantity is the
**differential** against an all-findings-absent reference, which isolates what this
patient's evidence did from what the circuit does to anything.

This is coupling *within the model*, never a causal claim about the patient, and the
report wording is tested to keep that distinction.

---

## 3. Two defects this work found and fixed

Both were live in the served path.

**Calibration was backend-blind.** `safety.npz` held a single temperature — the
classical one — so selecting the quantum backend scored the VQC's logits with a
constant fitted for a different model. Measured spread across backends: **1.89×**.
That silently distorts every probability, conformal set, and abstention threshold.
Fixed: per-backend calibration artifacts, preferred automatically by
`Calibration.load(backend=...)`.

**The shot schedule would commit on a clinically meaningless lead.** Without a
significance floor, QMBA committed to a top-1 margin of **6×10⁻⁷** — statistically
separated, because the shot noise happened to be smaller still. Statistical separation
and clinical significance are different questions. Fixed with a floor anchored to the
model's own calibration error (ECE 0.046–0.083, floor 0.05): *do not commit on a lead
finer than the resolution at which your probabilities are known to be accurate.*

---

## 4. What this is not

- **No quantum hardware.** Everything runs on PennyLane's `default.qubit` statevector
  simulator. The `QuantumDevice` seam accepts `lightning.qubit` or a Qiskit backend
  with no fusion-engine change, but that is a claim about the interface, not a result.
- **No quantum advantage.** An 8-qubit angle-encoded circuit over 8 hand-designed
  features is a small trainable classifier. Nothing here is faster or asymptotically
  better than classical. The claim is *native uncertainty*, not speedup.
- **Shot noise is not model error.** As `n_shots → ∞` the estimate converges to the
  analytic expectation, not to the truth. More measurement makes a confident wrong
  answer more confident. QMBA is a resolution budget, not a correctness guarantee, and
  it composes with — never replaces — the conformal and OOD abstention in
  `services/safety`.
- **The most interesting quantum design is still not wired.** `projection.py`
  (learned 1024-d → 8-qubit bottleneck) and the data re-uploading ansatz are complete
  and unused. Their barren-plateau reasoning is argued but not empirically
  demonstrated.

---

## 5. Where the code is

| File | What |
|---|---|
| `services/fusion/device.py` | Circuit definitions; `entangler="ring"\|"none"` |
| `services/fusion/quantum.py` | Serving VQC, shot-noise uncertainty, measurement entropy |
| `services/fusion/qmeasure.py` | Evidence-entanglement map, differential attribution |
| `services/fusion/qmba.py` | Measurement-budgeted abstention |
| `ml/evaluation/quantum_study.py` | The whole evidence pack, one artifact |
| `ml/training/recalibrate_backend.py` | Per-backend calibration |
| `tests/test_quantum_measurement.py` | 30 tests |

---

## 6. How to pitch this in 30 seconds

> Quantum is not doing the seeing — a CNN does that. Quantum does the deciding, and
> it gives us one thing no classical model has: the posterior's uncertainty comes from
> measurement physics, not from an ensemble we had to train. We turned that into a
> measurement budget. Easy patients resolve in 128 shots; hard ones get more; and when
> the budget cannot settle it, the system says *why* — 60 cases needed a human, 1
> needed a longer run. We also ran the control everyone should ask for: the same
> circuit with the entanglement removed. It scored better on log-likelihood. So we are
> not claiming entanglement wins — we are claiming we measured it.
