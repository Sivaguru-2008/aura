# AURA Baseline Comparison Report

> [!IMPORTANT]
> **What was and was not measured.** Every AURA row below is computed by this script
> from a served artifact and changes when the model changes. Every **competitor** row
> is a *published literature value*, cited to the table it was copied from and
> reproduced here for orientation only — nnU-Net and SwinUNETR were **not run** on this
> machine, this split, or this preprocessing. The two kinds of number are not comparable
> as a head-to-head, and §1 explains one specific reason they are not. Two rows that
> could not be traced to any published table (a "MONAI 3D ResU-Net" and a "ResNet-50
> linear probe") were removed on 2026-09-12 rather than kept under a softer label.

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
| nnU-Net | 0.8535 | 0.8895 | 0.8506 | 0.8203 | not measured | published: Isensee et al., "nnU-Net for Brain Tumor Segmentation", BrainLes 2020 (arXiv:2011.00848), abstract — BraTS 2020 winning entry, per-case 3-D |
| SwinUNETR | 0.8897 | 0.926 | 0.885 | 0.858 | not measured | published: Hatamizadeh et al., "Swin UNETR", BrainLes 2021 (arXiv:2201.01266), Table 3 — BraTS **2021** validation set (219 cases), per-case 3-D |

*Composite = mean of the three published region scores, computed here. SwinUNETR's
figures are on BraTS 2021, a larger and different cohort from the BraTS 2020 data AURA
was trained on. AURA's GPU latency is omitted rather than estimated: this evaluation
runs on CPU and no GPU timing was collected in this pass. See `docs/BENCHMARKS.md` §2
for the chest model's measured GPU numbers.*

---

## 2. Chest Radiograph Classification

| Model / Framework | Macro AUROC | Macro F1 | ECE | Inference Latency | OOD Safety | Source |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **AURA Chest (DenseNet-121)** | **0.8213** | **0.3853** | **0.1719** | **102.8 ms** | **Active** (rejects non-radiographs, OOD z-score) | measured here |

No chest comparator is tabulated here. The only external model this repo has actually
run is torchxrayvision's MIMIC DenseNet, as an independent label cross-check
(`artifacts/kappa_crosscheck.json`), and published CXR products report on different
finding sets and datasets — see `docs/MARKET_COMPARISON.md` §1 for that comparison and
its caveats.

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
> Classical PoE is the **served** backend. At this n the choice cannot rest on these
> rows; it rests on the preregistered external validation
> (`artifacts/chexpert_external.json`, n = 4,800 CheXpert studies), where
> classical PoE is more accurate than the VQC (0.5273 vs
> 0.4721, McNemar p < 0.001, Holm-adjusted). Per-class support here is
> thin (several classes in single digits), so per-class figures are directional only.
