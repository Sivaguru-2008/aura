# AURA Architectural Refactor — Mathematical Reference

Four safety-critical fixes to fusion, calibration, active inference and vision.
Each section gives the formulation actually implemented, a proof or derivation, and
an **honest caveat** section — because in a clinical decision-support system the
failure modes of a "fix" matter as much as its headline property. Every claim here
is exercised by the smoke tests referenced inline; none of it is aspirational.

> **Regulatory framing.** These are research-grade components. None is a validated
> medical device function. Coverage/again-guarantees below are *distribution-level*
> mathematical properties, not per-patient guarantees, and every module fails
> *toward abstention/wider sets*, never toward false confidence.

---

## Module 5 — Quantum-Classical Fusion & Barren-Plateau Mitigation

Files: `services/fusion/conflict.py` — the Wasserstein conflict guard — **is wired
into the serving path**: `services/fusion/engine.py` runs it on every fusion, and
`gateway/pipeline.py` feeds its resolved posterior to the safety engine, so the
diagnosis a clinician sees reflects the guard's decision.

> **Status note (design vs. wired).** `services/fusion/projection.py`
> (`JointProjection`) and `services/fusion/device.py::make_reuploading_qnode` are the
> **designed extension** for the high-dimensional embedding path (§5.1–5.2 below).
> They are complete and correct but **not yet wired into the serving pipeline**: the
> shipped VQC angle-encodes the hand-designed 8-channel `evidence.encode` vector via
> `device.make_qnode`, it does not route the 1024-d DenseNet embedding through the
> projection. The modules are marked EXPERIMENTAL in their own docstrings. §5.1–5.2
> describe the intended design, not current runtime behaviour.

### 5.1 Dimensionality bottleneck

The production vision backbone emits a 1024-d DenseNet embedding; the VQC has
`n_qubits = 8`. A trainable projection compresses the joint vector before encoding:

$$ \mathbf{x}_{\text{ang}} = \tanh(W\,\mathbf{x}_{\text{joint}} + \mathbf{b}), \quad W\in\mathbb{R}^{n_q\times d},\ \mathbf{x}_{\text{ang}}\in(-1,1)^{n_q}. $$

`tanh` bounds each feature to $(-1,1)$ so the encoder angle $\pi x_i \in (-\pi,\pi)$
— one non-aliased rotation, where $\partial\langle Z\rangle/\partial x$ is largest.
Glorot init keeps $\mathrm{Var}[W\mathbf{x}]\approx\mathrm{Var}[\mathbf{x}]$.

### 5.2 Why this attacks barren plateaus (and why re-uploading alone does not)

For an $n$-qubit circuit whose layers approximate a unitary 2-design, with a global
cost $C$, the gradient concentrates exponentially (McClean et al. 2018):

$$ \mathrm{Var}\!\left[\partial_\theta C\right] \in \mathcal{O}(2^{-n}). $$

Three levers act on this exponent; the design pins all three:

1. **Width $n$ is capped.** The projection fixes the encoded width at $n_q=8$
   *regardless of input dimension*. This is the only lever that touches the
   exponent directly — it is why the bottleneck, not the ansatz, is the primary
   mitigation.
2. **Local cost.** Readout is single-qubit $\langle Z_i\rangle$. Cerezo et al.
   (2021): for **local** cost functions on circuits of depth $\mathcal{O}(\log n)$,
   $\mathrm{Var}[\partial_\theta C]$ vanishes only **polynomially**, not
   exponentially. This is the actual guarantee we lean on.
3. **Data re-uploading.** `make_reuploading_qnode` re-encodes $\mathbf{x}$ each
   layer, interleaving $R_x,R_y,R_z$ rotations with a CNOT ring
   (Pérez-Salinas et al. 2020). This raises expressivity without deepening the
   trainable block and empirically keeps $\mathrm{Var}[\partial_\theta]$ off the
   floor.

> **Honest caveat.** Data re-uploading is a *mitigation*, **not a theorem**. There
> is no proof that re-uploading prevents barren plateaus; the exponential-in-$n$
> result is defeated here by (1) small $n$ and (2) the local cost, per Cerezo et
> al. Marketing the ansatz as the fix is wrong — the bottleneck and the local
> readout are. Separately: on a classical simulator a VQC of this size buys no
> representational power a small MLP lacks. Its justified role in AURA is as an
> *independent second estimator* whose disagreement with the PoE is a usable
> safety signal (§5.3), not as a superior classifier.

### 5.3 Wasserstein tie-breaker (conflict resolution)

The VQC posterior $p$ and the Bayesian product-of-experts posterior $q$ estimate
the same diagnosis distribution. Disagreement is measured with the **1-Wasserstein
distance on a clinical-severity axis** — because a categorical metric (KL, TV)
ignores *how dangerous* the disagreement is. Embed each diagnosis at severity
$s_i$; then with CDFs $F_p, F_q$ along that axis,

$$ W_1(p,q) = \int_{0}^{1} \bigl| F_p(t) - F_q(t) \bigr|\, dt = \min_{\gamma\in\Pi(p,q)} \sum_{i,j}\gamma_{ij}\,|s_i - s_j|. $$

Moving mass from `normal` ($s=0$) to `pneumothorax_dx` ($s=1$) yields $W_1\to 1$;
shuffling mass between adjacent-severity labels yields $W_1\to 0$. The general
ground-metric case is solved exactly as a transportation LP
(`distance_cost_matrix`, scipy `linprog`/HiGHS).

**Dynamic threshold.** $\tau_t = \max(\tau_{\text{base}},\ \mu_t + k\sigma_t)$ over a
rolling window of recent distances (EWMA-style). If $W_1 > \tau_t$: return the PoE
posterior and raise `high_epistemic`. This makes the fallback fire on *statistical
outliers* relative to the clinic's own recent disagreement level, not a hand-set
constant.

> **Live finding.** On the *currently shipped trained artifact* the guard fires on
> ordinary inputs ($W_1\approx0.55 \gg \tau=0.12$). That means the trained VQC and
> the Bayesian baseline genuinely diverge — evidence the VQC fit needs attention,
> and a concrete demonstration the guard is doing real work rather than rubber-
> stamping. Verified in `services/fusion/engine.py` end-to-end.

---

## Module 8 — Adaptive Conformal Inference under Non-Exchangeability

Files: `services/safety/aci.py`, persisted via `gateway/storage.py`
(`ConformalStateRow`, `OutcomeRow`, `record_outcome`), driven from the feedback
endpoint in `gateway/app.py`.

### 8.1 The problem

Split conformal's $1-\alpha$ coverage assumes **exchangeability**. Under covariate
shift the fixed $\hat q$ silently mis-covers.

### 8.2 Update rule (threshold parameterisation)

The canonical Gibbs–Candès (2021) update controls the miscoverage *level*
$\alpha_t$: $\ \alpha_{t+1}=\alpha_t+\gamma(\alpha-\mathrm{err}_t)$, with the set
built from the $(1-\alpha_t)$ quantile. We parameterise by the nonconformity
**threshold** $\hat q$ (keep class $c$ iff $s_c=1-p_c\le\hat q$; **larger $\hat q$ ⇒
wider set**). Raising the level lowers the threshold, so the equivalent update
carries the **opposite sign**:

$$ \boxed{\ \hat q_{t+1} = \hat q_t + \gamma\,\bigl(\mathbb{1}\{Y_t\notin \widehat C_t\} - \alpha\bigr)\ } \tag{$\star$} $$

Read directly: a **miss** ($\mathrm{err}=1$) gives $+(1-\alpha)>0$ → $\hat q$ rises →
sets widen; a **cover** gives $-\alpha<0$ → sets tighten.

> **Spec note.** The brief's formula $\hat q_{t+1}=\hat q_t+\gamma(\alpha-\mathrm{err})$
> is the *level*-space update; transcribing it verbatim onto a nonconformity
> threshold inverts the control loop and $\hat q$ runs to a rail (caught by the
> convergence test). ($\star$) is the threshold-space form with the sign made
> consistent. Both describe the same algorithm.

### 8.3 Coverage proof (no exchangeability needed)

Sum ($\star$) over $t=1..T$; it telescopes:

$$ \hat q_{T+1} - \hat q_1 = \gamma \sum_{t=1}^{T}\bigl(\mathrm{err}_t - \alpha\bigr). $$

$\hat q$ is self-correcting and stays in a bounded range $[q_{lo}, q_{hi}]$ (if
$\hat q$ is large every set contains all classes so $\mathrm{err}=0$ pulls it down;
if tiny, $\mathrm{err}=1$ pulls it up). Hence $|\hat q_{T+1}-\hat q_1|\le q_{hi}-q_{lo}$
and

$$ \left|\ \frac{1}{T}\sum_{t=1}^{T}\mathrm{err}_t - \alpha\ \right| \ \le\ \frac{q_{hi}-q_{lo}}{\gamma\,T}\ \xrightarrow[T\to\infty]{}\ 0. $$

Long-run empirical miscoverage → $\alpha$ **for any sequence**, adversarial or
shifting. Verified: under an injected shift the model degrades and set sizes
auto-grow from ≈1.0 to ≈3.9 while miscoverage holds at 0.10.

> **Honest caveat.** This is *marginal, long-run* coverage, not conditional or
> finite-sample. Choice of $\gamma$ trades adaptation speed vs. $\hat q$ variance
> ($\mathcal{O}(1/(\gamma T))$ bias vs. $\mathcal{O}(\gamma)$ noise). It needs a
> stream of *confirmed* outcomes; delayed/biased labelling weakens it. It does not
> fix a mis-specified score, only the threshold on it.

---

## Module 14 — Joint Expected Information Gain with Causal Masking

Files: `services/recommend/causal.py`, wired in `services/recommend/engine.py`.

### 14.1 Chain rule (exact identity)

$$ I(Y; X_i, X_j) = I(Y; X_i) + I(Y; X_j \mid X_i). $$

This is exact — the source of the "double counting" is scoring $I(Y;X_i)+I(Y;X_j)$
as if the conditional term equalled the marginal $I(Y;X_j)$.

### 14.2 Conditional term via covariance (the approximation)

For jointly-Gaussian $(Y,X_i,X_j)$ the information $X_j$ adds beyond $X_i$ shrinks
with the squared correlation, since $\rho_{ij}^2$ is the variance share of $X_j$
explained by $X_i$:

$$ I(Y; X_j \mid X_i) \ \approx\ I(Y; X_j)\,\bigl(1 - \rho_{ij}^2\bigr). $$

Exactly, for Gaussians $I(Y;X_j\mid X_i) = -\tfrac12\log(1-\rho_{Y j\cdot i}^2)$ with
$\rho_{Y j\cdot i}$ the **partial correlation**; the $(1-\rho_{ij}^2)$ form is the
first-order surrogate used for the $\mathcal{O}(KN)$ budget.

### 14.3 Causal gating and greedy selection

Redundancy is gated by a **hardcoded directed clinical graph** $M$ so deweighting
is causal, not merely correlational:

$$ r_{ij} = m_{ij}\,\rho_{ij}^2,\qquad m_{ij}\in[0,1]\ \text{iff edge } i\to j. $$

e.g. `cardiomegaly → bnp` ($m=0.8,\rho=0.8 \Rightarrow r=0.51$),
`consolidation → opacity` ($r=0.73$), `troponin → bnp` ($r=0.34$). Greedy marginal
gain of adding $j$ to selected set $S$ uses a cached novelty vector:

$$ \Delta(j\mid S) = I(Y;X_j)\prod_{i\in S}(1-r_{ij}) = I(Y;X_j)\cdot \nu_j. $$

$\nu_j$ is non-increasing as $S$ grows (**monotone, diminishing-returns /
submodular-style**); picking each marker updates $\nu$ in $\mathcal{O}(N)$, so $K$
steps cost $\mathcal{O}(K\cdot N)$ vs. the $\mathcal{O}(2^{N})$ outcome enumeration
it replaces. Verified: `troponin+bnp` joint EIG $0.83 < 1.0$ independent sum;
greedy diversifies away from a redundant second cardiac marker.

> **Honest caveat.** The $(1-\rho^2)$ surrogate and the hand-set graph/covariance
> are priors, **not** learned from this clinic — replace with partial correlations
> estimated from the local outcome log. MI feature-selection is only provably
> submodular under conditional-independence-given-$Y$; under strong synergy
> (XOR-like markers) greedy can be sub-optimal. The graph must be curated by a
> clinician: a wrong edge suppresses a genuinely informative test.

---

## Module 2 — 1-Channel Vision Conditioning & Feature Preservation

Files: `ml/vision_cxr/model.py` (`luminance_init_conv0`),
`ml/vision_cxr/losses.py` (`TotalVariationLoss`, `RegularizedMultiLabelLoss`),
wired in `ml/vision_cxr/train.py`, `validate.py`.

### 2.1 Luminance-weighted conv0 init

Present grayscale $g$ to the original 3-channel `conv0` by broadcasting, and filter
$o$ responds with $\sum_c W[o,c]*g = (\sum_c W[o,c])*g$ — the **plain channel sum**
the old code used. That treats R,G,B as equally luminance-relevant, inflating gain
and shifting the statistics BatchNorm was calibrated to. Instead collapse with the
BT.601 luma weights:

$$ W_{\text{new}}[o,0] = \alpha\,W[o,R] + \beta\,W[o,G] + \gamma\,W[o,B],\quad (\alpha,\beta,\gamma)=(0.299,0.587,0.114). $$

**Conservation property (verified to $5\times10^{-7}$):** for any grayscale $g$,
$W_{\text{new}}*g = W_{\text{orig}} * [\alpha g;\beta g;\gamma g]$ — the new 1-channel
filter reproduces the pretrained filter's response to the *luminance component* of
a colour image, conserving its tuned edge/texture selectivity. (Also fixes a latent
bug: the old code passed `bias=original_conv.bias` — a `Tensor`/`None` — where a
`bool` was expected.)

### 2.2 Total-Variation feature regularisation

Grad-CAM++ differentiates through the last conv map $F$; high-frequency activation
noise there yields speckled, ungrounded heatmaps. Penalise TV of $F$:

$$ \mathcal{L}_{\text{TV}}(F) = \lambda\sum_{b,k}\Bigl(\textstyle\sum_{i,j}|F_{i+1,j}-F_{i,j}| + |F_{i,j+1}-F_{i,j}|\Bigr)\big/Z $$

(anisotropic $\ell_1$; isotropic $\sqrt{d_x^2+d_y^2+\varepsilon}$ optional). This is a
piecewise-smoothness prior: activations stay flat within a region and change sharply
only at true structural boundaries, so attributions lock onto anatomy. Total loss
$\mathcal{L} = \text{BCE}(\text{logits},y) + \mathcal{L}_{\text{TV}}(F)$.

> **Honest caveat.** TV is a smoothness prior with a real cost: too large $\lambda$
> over-smooths and **erases small high-frequency lesions** — nodules, thin
> pneumothorax lines — the highest-severity findings. Default $\lambda=10^{-4}$;
> treat it as a tunable validated against *small-lesion recall*, never a free win.
> Luminance vs. plain-sum both "work"; the weighted form is better-motivated and
> keeps activation statistics closer to pretraining (less BN re-adaptation).

---

## References

- McClean et al., *Barren plateaus in quantum neural network training landscapes*, Nat. Commun. 2018.
- Cerezo et al., *Cost function dependent barren plateaus in shallow parametrized quantum circuits*, Nat. Commun. 2021.
- Pérez-Salinas et al., *Data re-uploading for a universal quantum classifier*, Quantum 2020.
- Gibbs & Candès, *Adaptive Conformal Inference Under Distribution Shift*, NeurIPS 2021.
- Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction*, 2023.
- Krause & Golovin, *Submodular Function Maximization*, 2014.
- Chattopadhay et al., *Grad-CAM++*, WACV 2018.
