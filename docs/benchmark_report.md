# AURA Baseline Comparison Report

> [!IMPORTANT]
> **What was and was not measured.** Every AURA row below is computed by this script
> from a served artifact and changes when the model changes. Every **competitor** row
> is a *published literature value*, cited to its source and reproduced here for
> orientation only — nnU-Net, SwinUNETR and MONAI were **not run** on this machine,
> this split, or this preprocessing. The two kinds of number are not comparable as a
> head-to-head, and §1 explains one specific reason they are not.

## 1. Brain MRI Segmentation — AURA measured vs. published baselines

> [!WARNING]
> **These Dice figures are not like-for-like.** AURA's Dice is pooled over
> **7,531 2-D axial slices**; the BraTS
> literature values below are **per-case 3-D** Dice, averaged over whole volumes.
> Pooled-2-D scoring flatters a model, because slices with no tumour are easy and
> numerous, and it never penalises through-plane inconsistency. Treat the AURA row as
> an internal regression metric, not as a BraTS leaderboard position. A comparable
> number requires per-case 3-D evaluation on the official validation set.

| Architecture | Mean Composite Dice | Whole Tumor | Tumor Core | Enhancing Tumor | CPU Latency (Study) | Source |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **AURA Brain (ResU-Net)** | **0.865** | **0.915** | **0.846** | **0.835** | **15.50 s** | measured here, pooled 2-D |
| nnU-Net | 0.871 | 0.920 | 0.852 | 0.841 | not measured | Isensee et al., *Nat. Methods* 18:203 (2021), BraTS20 per-case 3-D |
| SwinUNETR | 0.858 | 0.910 | 0.838 | 0.825 | not measured | Hatamizadeh et al., *MICCAI BrainLes* (2021), BraTS21 per-case 3-D |
| MONAI 3D ResU-Net | 0.835 | 0.895 | 0.812 | 0.798 | not measured | MONAI BraTS reference tutorial, per-case 3-D |

*AURA's GPU latency is omitted rather than estimated: this evaluation runs on CPU and
no GPU timing was collected in this pass. See `docs/BENCHMARKS.md` §2 for the chest
model's measured GPU numbers.*

---

## 2. Chest Radiograph Classification

| Model / Framework | Macro AUROC | Macro F1 | ECE | Inference Latency | OOD Safety | Source |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **AURA Chest (DenseNet-121)** | **0.8213** | **0.3853** | **0.1719** | **102.8 ms** | **Active** (rejects non-radiographs, OOD z-score) | measured here |
| ResNet-50 linear probe | 0.7650 | 0.2850 | 0.2850 | ~25 ms | None | indicative reference, **not measured here** — no uncalibrated ResNet-50 probe is trained in this repo |

---

## 3. Cross-Modal Evidence Fusion — measured, n = 69

All rows read from `artifacts/benchmark.json`; both backends are temperature-scaled on
their own calibration split so ECE is not inflated for either.

| Backend | Accuracy | NLL | ECE | Brier | Macro AUROC | Conformal Coverage (target 90%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical PoE** | **0.6957** | 1.0577 | 0.2194 | 0.4857 | 0.7875 | 91.3% |
| **Quantum VQC (8-qubit)** | 0.6377 | 1.2123 | 0.2381 | 0.5699 | 0.7696 | 92.8% |

> [!NOTE]
> **The gap between these two backends is not statistically resolvable at n = 69.**
> 48/69 correct vs 44/69 is a
> difference of 4 cases. The 95% intervals are
> [0.581, 0.795] and
> [0.520, 0.744] — overlapping across nearly
> their whole range — and a paired McNemar test cannot reach significance under *any*
> assignment of the discordant pairs (best case p = 0.125).
> Classical PoE is served as the fair-accuracy reference on grounds of interpretability
> and cost, **not** on a demonstrated accuracy advantage. Per-class support is thin
> (several classes in single digits), so per-class figures are directional only.
