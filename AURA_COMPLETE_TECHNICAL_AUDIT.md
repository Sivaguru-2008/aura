# AURA Complete Technical Audit & Architecture Roadmap
**System Tagline**: *Adaptive Uncertainty-aware Reasoning Assistant* — "The clinical copilot that knows what it doesn't know."

---

## Section 1: Project Overview

### What the System Is
AURA (Adaptive Uncertainty-aware Reasoning Assistant) is a clinical diagnostic copilot designed around the core principle of **calibrated doubt**. Traditional medical AI models act as black-box classifiers, outputting static probabilities that fail silently when exposed to out-of-distribution (OOD) images or highly ambiguous clinical cases. AURA's primary output is not just a diagnosis, but a statistical measure of its own uncertainty—identifying missing evidence, explaining its decisions, bounding its predictions with conformal coverage guarantees, and explicitly abstaining from a clinical claim when it is unsafe to make one.

AURA operates 100% offline on the edge. It runs without cloud connections, external API dependencies, or protected health information (PHI) leaving the local machine. 

### How It Works & Main Workflow
AURA accepts medical imaging studies (frontal chest radiographs or volumetric brain MRIs) and patient clinical metadata (symptoms, lab results, patient history). The intake data is processed through an intelligent routing and modality detection layer, which dispatches it to the corresponding clinical engine (AURA Thorax or AURA NeuroMind).

```
[ Upload Staged ] ──► [ Modality Router ] ──► [ Modality Engine Dispatch ]
                                                    │
             ┌──────────────────────────────────────┴──────────────────────────────────────┐
             ▼                                                                             ▼
    [ AURA Thorax (CXR) ]                                                         [ AURA NeuroMind (MRI) ]
             │                                                                             │
      DenseNet-121 Vision                                                           MRI Foundation Layer
             │                                                                             │
     Evidence Encoder                                                               3D ResU-Net Segmenter
             │                                                                             │
     Quantum VQC Fusion ◄──► Conflict Guard                                         Calibrated Presence
             │                                                                             │
      Clinical Reasoner                                                             Neuro Safety Engine
             │                                                                             │
     Safety Engine Assess                                                            Report Grounding
             │                                                                             │
             └──────────────────────────────────────┬──────────────────────────────────────┘
                                                    ▼
                                    [ Grounded Report & Dashboard ]
                                                    │
                                                    ▼
                                     [ Clinician Feedback & ACI ]
```

1. **Intake & Routing**: Uploaded assets (DICOM, NIfTI, NRRD, ZIP, PNG, or JPG) are analyzed by the [ModalityRouter](file:///e:/AURA/aura-main/aura/backend/core/router/router.py). It reads DICOM headers or extracts pixel geometry features to verify if the study is a chest X-ray or a brain MRI, rejecting unsupported modalities.
2. **Vision/Preprocessing Engine**:
   - **Thorax (Chest X-Ray)**: The image is normalized and passed through a fine-tuned [DenseNet-121 model](file:///e:/AURA/aura-main/aura/ml/vision_cxr/model.py) to predict 7 clinical findings.
   - **NeuroMind (Brain MRI)**: The volumetric MRI is standardized through the [MRI Foundation Pipeline](file:///e:/AURA/aura-main/aura/backend/foundation/mri/pipeline.py), performing RAS reorientation, bias field correction, brain masking, and resampling. The standardized volume is segmented by a 3D/2D [Residual U-Net](file:///e:/AURA/aura-main/aura/backend/vision/brain/model/network.py).
3. **Evidence Encoding**: For Thorax, vision finding scores and structured clinical priors are encoded into a compact, normalized 8-channel evidence vector.
4. **Evidence Fusion**: The 8-channel vector is passed to the [FusionEngine](file:///e:/AURA/aura-main/aura/services/fusion/engine.py), which uses an 8-qubit Variational Quantum Circuit (VQC) in PennyLane to calculate expectation values, mapping them to 6 diagnostic logits. A Product-of-Experts (PoE) model runs in parallel, and a **Wasserstein Conflict Guard** falls back to PoE if the quantum circuit departs from classical sanity.
5. **Clinical Reasoning**: The [ClinicalReasoner](file:///e:/AURA/aura-main/aura/services/reasoning/engine.py) adjusts the fusion posterior by applying likelihood ratios from clinical guidelines (e.g., ACC/AHA, IDSA/ATS) matching the patient's symptoms and labs (e.g., BNP, WBC, fever).
6. **Safety & Calibration**: The [SafetyEngine](file:///e:/AURA/aura-main/aura/services/safety/engine.py) applies temperature scaling, computes Mondrian conformal prediction sets, scores OOD energy, and triggers abstentions if safety margins are violated.
7. **Explainability & Recommendations**: The system generates Grad-CAM++ overlays, leave-one-out attributions, and computes the Expected Value of Information (EVOI) to recommend the next diagnostic action.
8. **Report Generation**: The [ReportEngine](file:///e:/AURA/aura-main/aura/services/report/engine.py) drafts a report with sentences grounded in evidence nodes.
9. **Clinician Feedback Loop**: The doctor signs off on the case. Their feedback writes back to the local database, feeding the **Adaptive Conformal Inference (ACI)** loop to adjust prediction bounds under distribution shift.

---

## Section 2: Repository Structure

AURA's codebase is structured inside the package root [`aura/`](file:///e:/AURA/aura-main/aura). Here is the layout and purpose of each major directory:

### Directory Tree & Responsibilities

- [`aura/backend/`](file:///e:/AURA/aura-main/aura/backend/): Represents the new clinical operating system routing and volumetric brain MRI layer.
  - [`core/`](file:///e:/AURA/aura-main/aura/backend/core/): Handles HTTP upload intake, validation rules (aspect ratio, size limits), logging, error taxonomies, and the modality router.
    - [`router/`](file:///e:/AURA/aura-main/aura/backend/core/router/): Detects modality from DICOM tags or pixel geometries.
  - [`foundation/mri/`](file:///e:/AURA/aura-main/aura/backend/foundation/mri/): Preprocessing foundation for 3D MRIs. Contains DICOM, NIfTI, and NRRD loaders, reorientation routines, N4 bias field slots, quality checks, spatial registration tools, and coordinate geometry.
  - [`vision/brain/`](file:///e:/AURA/aura-main/aura/backend/vision/brain/): ResU-Net segmentation model training and inference routines. Includes custom 3D convolutions, Multi-Task heads (Presence, Size, Quality, Embedding, Segmentation), datasets, metrics, and loss functions.
  - [`engines/neuro/`](file:///e:/AURA/aura-main/aura/backend/engines/neuro/): The NeuroMind modality engine implementing the standard `AnalysisEngine` contract.
  - [`api/`](file:///e:/AURA/aura-main/aura/backend/api/): FastAPI route definitions for modality routing and multi-engine processing.
  - [`models/`](file:///e:/AURA/aura-main/aura/backend/models/): Pydantic wire contracts for routing envelopes.
  - [`bootstrap.py`](file:///e:/AURA/aura-main/aura/backend/bootstrap.py): Mounts the new routing backend onto the gateway app.
- [`aura/services/`](file:///e:/AURA/aura-main/aura/services/): Houses the core clinical intelligence modules, historically designed around the chest radiograph pipeline.
  - [`vision/`](file:///e:/AURA/aura-main/aura/services/vision/): Contains the chest X-ray intake gate (`xray_gate.py`) and the DenseNet-121 inference engine (`engine.py`).
  - [`fusion/`](file:///e:/AURA/aura-main/aura/services/fusion/): Houses the VQC PennyLane circuit (`quantum.py`, `device.py`), Product-of-Experts (`classical.py`), attention-gated neural fusion (`learnable.py`), and the Wasserstein Conflict Guard (`conflict.py`).
  - [`reasoning/`](file:///e:/AURA/aura-main/aura/services/reasoning/): Combines patient-specific labs/symptoms with clinical guidelines using likelihood ratios.
  - [`safety/`](file:///e:/AURA/aura-main/aura/services/safety/): Calibrates probabilities, generates conformal sets, detects OOD energy, and handles ACI.
  - [`explain/`](file:///e:/AURA/aura-main/aura/services/explain/): Calculates Grad-CAM/++, Integrated Gradients, SmoothGrad, and Occlusion maps.
  - [`recommend/`](file:///e:/AURA/aura-main/aura/services/recommend/): Decision-theoretic EVOI and EIG engine, utilizing causal dependency graphs to deweight redundant tests.
  - [`report/`](file:///e:/AURA/aura-main/aura/services/report/): structured findings to grounded clinician-style reports.
  - [`memory/`](file:///e:/AURA/aura-main/aura/services/memory/): In-memory similarity search over anatomical embeddings.
  - [`agent/`](file:///e:/AURA/aura-main/aura/services/agent/): Active diagnostic agent running sequential uncertainty minimization loops.
- [`aura/gateway/`](file:///e:/AURA/aura-main/aura/gateway/): FastAPI gateway orchestration. Defines routes (`app.py`), in-process event bus (`pipeline.py`), and database storage transactions (`storage.py`).
- [`aura/schemas/`](file:///e:/AURA/aura-main/aura/schemas/): Pydantic models acting as the strict interface boundaries between engines.
- [`aura/common/`](file:///e:/AURA/aura-main/aura/common/): Shared configuration (`config.py`), event bus logic (`eventbus.py`), and mathematical utilities (`mathx.py`).
- [`aura/ml/`](file:///e:/AURA/aura-main/aura/ml/): MIMIC-CXR dataset loaders and training scripts for both the vision and fusion networks.
- [`aura/tests/`](file:///e:/AURA/aura-main/aura/tests/): The unit and integration test suite, containing 135+ tests.

### Key Data Flows

1. **Intake to Routing**: Uploaded bytes are written to a temp file, hashed (SHA-256), and passed to the router. The router uses `pydicom` to extract tags or reads the pixel array to verify format.
2. **Modality Engine Analysis**: 
   - Chest X-rays flow: `gateway/pipeline.py` -> `services/vision` (grayscale checks, aspect checking, crop to 224x224, forward pass on DenseNet-121) -> `services/fusion` (evidence encoding vector, quantum VQC execution, conflict guard) -> `services/reasoning` -> `services/safety` -> `services/explain` / `services/recommend` -> `services/report` -> `gateway/storage.py` (persistent SQLite state) -> JSON/HTML response to SPA dashboard.
   - Brain MRIs flow: `gateway/pipeline.py` -> `backend/engines/neuro/engine.py` (validate NIfTI/DICOM/NRRD format, complete sequence checks) -> `backend/foundation/mri` (reorientation, N4 bias correction, masking, standardization) -> `backend/vision/brain/inference.py` (Curriculum-based 3D ResU-Net multi-task forwarding) -> `backend/engines/neuro/bundle.py` (construct CaseBundle, compute volume metrics, select display slice) -> `gateway/storage.py` -> JSON response.

---

## Section 3: Chest X-Ray Pipeline

AURA Thorax is a fully-wired clinical pipeline designed for the analysis of frontal chest radiographs.

### Dataset
The pipeline is trained and validated on the **MIMIC-CXR** dataset. The validation manifest [`mimic_cxr_aug_validate.csv`](file:///e:/AURA/aura-main/datasets/) contains $n=2,099$ patient-disjoint, resolved frontal radiographs. Target labels map 7 image findings and 6 primary clinical diagnoses extracted from radiology reports using CheXpert rule-based parsing.

### Model
The vision backbone is a **DenseNet-121** architecture ([`DenseNet121CXR`](file:///e:/AURA/aura-main/aura/ml/vision_cxr/model.py)). The standard model is adapted for medical imaging: the first convolutional layer is modified to accept a single grayscale channel instead of three RGB channels, and the final classification head is replaced with a 7-way multi-label logistic output.

### Training
Trained using Multi-Label Binary Cross BCE loss, utilizing the Adam optimizer with weight decay. Checkpoints are Promoted from epoch 7 weights, achieving a **macro-AUROC of 0.821** (95% CI: 0.811–0.832) on the validation split.

### Inference
Uploaded images pass through [`xray_gate.py`](file:///e:/AURA/aura-main/aura/services/vision/xray_gate.py), checking for correct modality, aspect ratios, and color profiles. Valid grayscale images are cropped and resized to $224 \times 224$ pixels using area-averaging interpolation to avoid spatial aliasing.

### Calibration
Raw sigmoid scores undergo Platt scaling. Operating thresholds and logistic parameters are fitted on the validation split ($n=2,099$) to minimize the Expected Calibration Error (ECE). Calibration reduces the ECE to **0.023**, guaranteeing that predicted probabilities match true pathological prevalence.

### Fusion
The 7 vision finding probabilities are combined with patient-specific priors (smoker, prior cancer, fever, age, immunocompromised) into an 8-channel evidence vector $x \in [0, 1]^8$. This vector is passed to the [FusionEngine](file:///e:/AURA/aura-main/aura/services/fusion/engine.py):
- **VQC (Quantum)**: Translates $x$ into RY rotation angles on 8 qubits, entangles them through a CNOT ring, and computes expectation values $\langle PauliZ \rangle$ mapped to 6 diagnostic logits.
- **Product of Experts (Classical)**: Models channels as independent experts, summing their log-likelihoods.
- **Conflict Guard**: A Wasserstein distance metric on a 1D clinical severity scale ($W_1(p_{VQC}, p_{PoE})$) is evaluated. If EMD exceeds a dynamic threshold ($\tau_t$, EWMA of recent case distance + $3 \sigma$), the system falls back to PoE, raising a `high_epistemic` conflict flag.

### Clinical Report
Composed by the `ReportEngine` and formatted by `clinical_report.py`. It presents five grounded sections: Findings, Impression, Recommendations, Differential, and Confidence.

### Explainability
Calculated dynamically for the leading finding:
- **Grad-CAM & Grad-CAM++**: Localizes conv feature map activations weighted by gradients.
- **Integrated Gradients**: Axiomatic attribution integrating path gradients from a black baseline.
- **SmoothGrad**: Averages gradients over Gaussian-perturbed inputs to reduce noise.
- **Occlusion**: Slides an 12px grey patch, measuring probability drops.

### Confidence Estimation
Consists of three layers:
- **Epistemic Variance**: Measures mutual information between predictions from a $k$-member deep ensemble (fits of bootstrapped training data). Falls back to input perturbation variance if the ensemble is absent.
- **Conformal Sets**: Evaluates Mondrian (class-conditional) conformal thresholds at 90% confidence, ensuring the true diagnosis is bounded.
- **OOD Energy**: Computes logit energy $E(x) = -T \log \sum e^{f_i(x)/T}$. A z-score is calculated against in-distribution statistics.

### Strengths
- Calibrated probabilities (ECE: 0.023).
- Strict safety loop resolving quantum decisions with classical PoE guards.
- Adaptive Conformal Inference (conformal threshold updates from clinician feedback).
- Grounded reports with direct provenance tracking.

### Weaknesses & Limitations
- **Fixed Vocabulary**: Limited to 7 findings and 6 diagnoses; incidental findings (like lines, tubes, or hiatal hernia) are not modeled.
- **In-Memory Memory Index**: Search embeddings are not saved across service restarts.
- **UI Bounding Boxes**: The web dashboard renders static anatomical bounding boxes rather than coordinates derived from Grad-CAM++.

---

## Section 4: Brain MRI Pipeline

AURA NeuroMind is a volumetric MRI processing and segmentation pipeline built to analyze brain tumors.

### Upload
Supports volumetric MRI uploads as a DICOM series directory, NIfTI (.nii or .nii.gz) files, or NRRD volumes. Uploads can be packaged in ZIP archives.

### Validation
[`NeuroMindEngine.validate_input`](file:///e:/AURA/aura-main/aura/backend/engines/neuro/engine.py#L159) checks if the file is a readable volumetric format. It enforces sequence completeness: the model requires all four standard sequences (**FLAIR, T1, T1CE, and T2**). It also verifies that the series contains more than 1 slice to reject 2D image exports.

### Transport Guard
If a user attempts to upload a 2D image export (PNG, JPEG, TIF), the engine rejects the image with a clinical disclosure showing the performance penalty of single-sequence models:
> Single-sequence whole-tumor Dice drops to 0.52 (FLAIR), 0.28 (T2), 0.02 (T1), and 0.00 (T1ce) compared to 0.58 when utilizing all four sequences.

### Preview
The system selects a representative 2D display slice by scanning the segmentation mask and picking the slice containing the highest volume of tumor tissue, ensuring that saliency overlays do not render on blank slices.

### Sequence Detection
The [RuleBasedSequenceDetector](file:///e:/AURA/aura-main/aura/backend/foundation/mri/sequence.py) reads DICOM Series Description tags or scans pixel geometry patterns to classify input sequences into T1, T1CE, T2, FLAIR, DWI, ADC, SWI, or PD.

### Volume Stacking
If sequences are uploaded as separate volumes, the [MRIIntakeManager](file:///e:/AURA/aura-main/aura/backend/foundation/mri/intake_manager.py) stacks them into a 4D array ($C, H, W, D$) and registers their channel order.

### Preprocessing
Managed by the `MRIFoundationPipeline`:
1. **Reorientation**: Standardizes voxel axes to canonical RAS orientation.
2. **N4 Bias Correction**: Removes scanner-induced RF field intensity inhomogeneities (N4BiasFieldCorrection slot).
3. **Brain Masking**: Computes an Otsu-based brain envelope to isolate parenchymal pixels.
4. **Isotropic Resampling**: Resamples voxel sizes to a uniform $1.0\text{ mm}^3$ spacing.
5. **Normalisation**: Normalizes intensities using min-max scaling or z-score normalization.

### Segmentation
The [BrainVisionNetwork](file:///e:/AURA/aura-main/aura/backend/vision/brain/model/network.py) is a 3D/2D Residual U-Net trained on the **BraTS2020** dataset (composite validation Dice: **0.875**). It segments three tumor components:
- Necrotic and non-enhancing tumor core.
- Peritumoral edema.
- Enhancing tumor.

### Classification
A Platt-calibrated **Presence Head** on the encoder bottleneck calculates the probability of tumor presence.

### Visualization
The [NeuroView](file:///e:/AURA/aura-main/aura/backend/engines/neuro/neuroview.py) service maps segmented regions (WT, TC, ET) to color-coded canvas overlays on the display slice.

### Reporting
Generates a grounded report detailing:
- Calibrated tumor presence probability.
- Calculated tumor burden volumes in voxels and cubic millimeters ($\text{mm}^3$).
- Technical warnings (OOD energy, motion artifacts, quality scores).

### Explainability
Exposes single-sequence Dice degradation statistics directly in the caveats to communicate model limitations.

### Strengths
- High composite Dice (0.875) on BraTS gliomas.
- Preprocessing standardizes scanners and protocols.
- Enforces multi-sequence completeness.

### Weaknesses & Limitations
- **Subtype Blindness**: The model does not classify tumor subtypes (e.g., meningioma, metastasis). It outputs a general `Diagnosis.BRAIN_TUMOR` finding.
- **Preoperative Bias**: Only validated for preoperative cases; postoperative scans or radiation necrosis can confuse the segmentation.
- **CT/MRI Ambiguity**: Pixel geometry cannot distinguish axial head CTs from MRIs; headerless CT scans are routed as MRIs at capped confidence.

---

## Section 5: Quantum Stack Analysis

AURA integrates PennyLane variational quantum circuits into its fusion, safety, and decision-support pipelines. Here is an audit of every quantum component currently in the repository.

### Implemented Quantum Components

#### 1. Quantum Serving Fusion Backend
- **Location**: [`aura/services/fusion/quantum.py`](file:///e:/AURA/aura-main/aura/services/fusion/quantum.py)
- **Purpose**: Serves the trained VQC parameters, computes expected PauliZ values, and evaluates posterior probabilities.
- **Inputs**: 8-channel normalized evidence vector $x \in [0, 1]^8$.
- **Outputs**: 6-class diagnosis posterior probabilities, Monte Carlo shot-noise standard deviations ($std\_d$), and logit arrays.
- **Current Usage**: Invoked by the `FusionEngine` to perform multi-modal evidence fusion for Thorax studies.
- **Contributions**:
  - *Clinical*: Native uncertainty propagation (MC shot noise) modeled on hardware measurement physics.
  - *Engineering*: Pure-numpy execution path for fast CPU inference.
  - *Hackathon*: Demonstrates a working VQC compared to classical heads.
  - *Scientific*: Implements analytic expectations paired with finite-shot variance simulation.
- **Classification**: **Production Ready** (Fully integrated and evaluated).

#### 2. Variational Quantum Circuit (QNode)
- **Location**: [`aura/services/fusion/device.py::make_qnode`](file:///e:/AURA/aura-main/aura/services/fusion/device.py#L14)
- **Purpose**: Defines the VQC tape structure using PennyLane.
- **Inputs**: Batch of evidence vectors $x$, parameter weight tensor $\theta$ (rotations).
- **Outputs**: List of 8 expectation values $\langle PauliZ(i) \rangle$.
- **Current Usage**: Shared circuit structure for both VQC training and serving paths.
- **Circuit Architecture**:
  - *Encoding*: Angle-encodes $x_i$ as $RY(\pi \cdot x_i)$ on qubit $i$.
  - *Ansatz*: $n\_layers$ of trainable $RY(\theta_{l,i,0})$ and $RZ(\theta_{l,i,1})$ rotations per qubit.
  - *Entangler*: Ring of CNOT gates entangling qubit $i$ with $(i+1)\pmod 8$.
- **Contributions**:
  - *Clinical*: Captures high-order interactions in the Hilbert space.
  - *Engineering*: Supports parameter broadcasting for fast batch simulation.
  - *Hackathon*: Clear ablation control (parameter `entangler="none"`) to measure entanglement utility.
  - *Scientific*: Provides a controlled baseline to isolate the impact of entanglement.
- **Classification**: **Production Ready**.

#### 3. Quantum Measurement-Budgeted Abstention (QMBA)
- **Location**: [`aura/services/fusion/qmba.py`](file:///e:/AURA/aura-main/aura/services/fusion/qmba.py)
- **Purpose**: Sequentially allocates shot budgets to resolve decisions using statistical margins.
- **Inputs**: Evidence vector $x$.
- **Outputs**: `BudgetDecision` (committed flag, spent shots, margin standard deviation, limiting factor).
- **Current Usage**: Provides a sequential decision framework for VQC inference.
- **Logic**: Starts with 128 shots, projects logits, and estimates the decision margin:
  $$margin = p_{top1} - p_{top2}$$
  Calculates margin variance from shot noise. If the margin is separated from zero by $z \ge 2.0$, it commits. Otherwise, it doubles the shot budget up to 8,192 shots. If it reaches the ceiling without separation, it abstains, categorizing the case as *measurement-limited* or *model-limited*.
- **Contributions**:
  - *Clinical*: Distinguishes between cases needing more measurement versus cases requiring human escalation.
  - *Engineering*: Minimizes shot cost on simulators and hardware.
  - *Hackathon*: Highly novel application of quantum measurement physics.
  - *Scientific*: Shows that shot allocation can scale dynamically based on input ambiguity.
- **Classification**: **Production Ready**.

#### 4. Evidence Entanglement Mapping (Q-Correlator)
- **Location**: [`aura/services/fusion/qmeasure.py`](file:///e:/AURA/aura-main/aura/services/fusion/qmeasure.py)
- **Purpose**: Computes connected two-qubit correlators to measure how the circuit couples findings.
- **Inputs**: Evidence vector $x$, reference vector $x_{ref}$ (all findings zero).
- **Outputs**: Symmetric correlation matrix, differential correlation matrix ($C(x) - C(x_{ref})$), and measurement entropy.
- **Logic**: Evaluates:
  $$C_{ij} = \langle Z_i Z_j \rangle - \langle Z_i \rangle \langle Z_j \rangle$$
  Subtracts the baseline correlation of the empty state $x_{ref}$ to isolate the patient-attributable coupling.
- **Contributions**:
  - *Clinical*: Visualizes the patient-specific diagnostic relationships captured by the circuit.
  - *Engineering*: Extracts single-qubit and two-qubit expectations in a single compilation pass.
  - *Hackathon*: Provides an entanglement interpretability tool for medical AI.
  - *Scientific*: Disproves the use of raw correlators, establishing differential baseline subtraction.
- **Classification**: **Production Ready**.

#### 5. Joint Feature Projection
- **Location**: [`aura/services/fusion/projection.py`](file:///e:/AURA/aura-main/aura/services/fusion/projection.py)
- **Purpose**: Compresses high-dimensional vision embeddings to qubit rotation angles.
- **Inputs**: 1024-d DenseNet feature vector + prior vector.
- **Outputs**: 8-dimensional compressed vector in $(-1, 1)$.
- **Logic**: Trains a linear layer followed by a $Tanh$ activation:
  $$x_{proj} = \tanh(W x + b)$$
  bounding the output to $(-1, 1)$ to keep rotation angles within $(-\pi, \pi)$.
- **Contributions**:
  - *Scientific*: Proposes a joint classical-quantum boundary to mitigate barren plateaus by keeping the qubit width small.
- **Classification**: **Experimental** (Designed and written, but not imported in served paths).

#### 6. Data Re-uploading Ansatz
- **Location**: [`aura/services/fusion/device.py::make_reuploading_qnode`](file:///e:/AURA/aura-main/aura/services/fusion/device.py#L99)
- **Purpose**: Hardware-efficient data re-uploading circuit designed to mitigate barren plateaus.
- **Inputs**: Feature vector $x$, weight tensor $\theta$.
- **Outputs**: Qubit expectations.
- **Logic**: Re-injects features $x$ at the start of each layer:
  $$\text{Layer} = [RX(\pi x), RY(\pi x), RZ(\pi x)] \to [RX(\theta), RY(\theta), RZ(\theta)] \to \text{CNOT ring}$$
- **Classification**: **Experimental** (Not wired into live inference pipelines).

---

## Section 6: What Quantum Is Still Missing

Analyzing the repository from the perspective of a Quantum Hackathon Judge, several quantum technologies could realistically improve the project's clinical capabilities without reducing the accuracy of the classical backbones.

```
       [ Classical Vision Backend ]                  [ Classical Preprocessing ]
        (DenseNet-121 / 3D ResU-Net)                        (MRI Foundation)
                     │                                             │
                     ▼                                             ▼
       ┌───────────────────────────┐                 ┌───────────────────────────┐
       │  Quantum Autoencoder /    │                 │  Quantum Attention        │
       │  Quantum Kernel Learning  │                 │  in 3D Latent bottleneck  │
       └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                     │                                             │
                     ▼                                             ▼
       ┌───────────────────────────┐                 ┌───────────────────────────┐
       │  Quantum Bayesian Net     │                 │  Quantum Longitudinal     │
       │  (Clinical Guideline LR)  │                 │  Progression Modeling     │
       └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                     │                                             │
                     └──────────────────────┬──────────────────────┘
                                            ▼
                             [ Dynamic Decision / Reporting ]
```

### Missing Quantum Components

1. **Quantum Feature Autoencoder (QAE)**: Compresses the high-dimensional classical embeddings (1024-d from DenseNet-121 or 128-d from the brain U-Net bottleneck) into a low-dimensional quantum state. This replaces classical linear PCA or heuristic projections, preserving information using unitary operations.
2. **Quantum Kernel Learning (QKL) & QSVM**: Projects compressed features into a high-dimensional Hilbert space, calculating a quantum kernel matrix $K_{ij} = |\langle \psi(x_i) | \psi(x_j) \rangle|^2$. This can improve decision boundaries for rare pathologies (e.g., Pneumothorax or Glioma subtypes) where linear classifiers perform poorly.
3. **Quantum Attention Mechanism (Q-Attention)**: Replaces self-attention layers in the 3D U-Net encoder-decoder bottleneck. Using quantum circuits to compute query-key overlaps in Hilbert space can capture long-range spatial correlations in volumetric MRI scans.
4. **Quantum Bayesian Networks (QBN)**: Replaces the classical `ClinicalReasoner` rule-engine. It models guideline likelihood ratios using joint probability amplitudes, allowing medical variables (e.g., BNP, fever, consolidation) to interact natively without assuming classical independence.
5. **Quantum Causal Inference (Q-Causal)**: Extends the causal graph in `RecommendEngine` to model causal relationships as directed quantum channels, identifying confounding factors in multimodal diagnostics.
6. **Quantum Longitudinal Progression Modeling**: Tracks brain tumor growth or regression across time series. Progression can be modeled as a unitary trajectory in Hilbert space, projecting future volume states based on current parameters.
7. **Quantum Federated Learning (QFL)**: Enables secure, collaborative training of chest and brain models across clinical institutions by sharing local quantum gradients instead of raw pixel files or patient data.
8. **Quantum Workflow Optimization (QAOA)**: Solves the clinical scheduling problem (prioritizing high-acuity studies, scheduling MRI scanner time, and assigning radiologists) using the Quantum Approximate Optimization Algorithm.

---

## Section 7: Integration Plan

This plan details how the missing quantum components can be integrated into AURA, prioritizing **augmentation** over replacement to safeguard classical performance.

### 1. Quantum Autoencoder (QAE) for DenseNet Compression
- **Why**: Replaces the classical projection boundary, reducing the 1024-d DenseNet feature vector to 8 qubits without losing higher-order covariance.
- **Where to Connect**: Connects between [`services/vision/engine.py`](file:///e:/AURA/aura-main/aura/services/vision/engine.py) and the `FusionEngine`.
- **Inputs**: 1024-d DenseNet feature embedding.
- **Outputs**: 8-dimensional normalized angle vector.
- **Files to Modify**: Create `services/fusion/qae.py`; modify `services/fusion/engine.py` to route through the QAE.
- **Expected Cost**: Medium simulator overhead (requires 18 qubits to simulate encoder-decoder states).
- **Expected Benefit**: Retains more spatial and semantic information than linear projections, improving downstream VQC accuracy.
- **Risk**: Optimization instability during training.
- **Performance Impact**: Augments the pipeline; does not affect classical PoE or DenseNet validation performance.

### 2. Quantum Kernel Learning (QKL) for Tumor Subtype Classification
- **Why**: Enables the NeuroMind engine to classify tumor subtypes (e.g., Glioma vs. Metastasis vs. Meningioma) using high-dimensional classification boundaries.
- **Where to Connect**: Connects to the bottleneck output of the brain network in [`backend/vision/brain/inference.py`](file:///e:/AURA/aura-main/aura/backend/vision/brain/inference.py).
- **Inputs**: 128-d latent representation from the U-Net encoder.
- **Outputs**: Subtype classification probabilities.
- **Files to Modify**: Create `backend/engines/neuro/qkl.py`; modify `backend/engines/neuro/engine.py` to add subtype predictions.
- **Expected Cost**: High simulator cost (calculating the kernel matrix scales quadratically with batch size: $O(N^2)$).
- **Expected Benefit**: Accurate classification of rare tumor types from small training cohorts.
- **Performance Impact**: Augments the pipeline; U-Net segmentations remain unaffected.

### 3. Quantum Bayesian Network (QBN) for Clinical Reasoning
- **Why**: Models joint dependencies between symptoms, labs, and imaging findings without assuming classical independence.
- **Where to Connect**: Replaces the log-linear addition in [`services/reasoning/engine.py`](file:///e:/AURA/aura-main/aura/services/reasoning/engine.py).
- **Inputs**: Imaging posteriors from the fusion engine + multimodal context (BNP, WBC, fever).
- **Outputs**: Adjusted multi-modal posterior.
- **Files to Modify**: `services/reasoning/engine.py`.
- **Expected Cost**: Low simulator cost (requires 4 to 6 qubits representing key variables).
- **Expected Benefit**: Captures complex, non-linear dependencies (e.g., how BNP elevates in the presence of both effusion and cardiomegaly).
- **Performance Impact**: Augments the pipeline; runs in parallel with the classical reasoner.

### 4. Quantum Attention in 3D U-Net Bottleneck
- **Why**: Captures long-range 3D spatial correlations in brain volumes.
- **Where to Connect**: Integrated inside the encoder bottleneck block of [`backend/vision/brain/model/blocks.py`](file:///e:/AURA/aura-main/aura/backend/vision/brain/model/blocks.py).
- **Inputs**: 3D feature tensor.
- **Outputs**: Attention-weighted 3D feature tensor.
- **Files to Modify**: `backend/vision/brain/model/blocks.py`, `backend/vision/brain/model/network.py`.
- **Expected Cost**: Extremely high (requires pocket-sized quantum circuits run per spatial pixel patch).
- **Expected Benefit**: Improves segmentation accuracy around diffuse tumor boundaries.
- **Performance Impact**: Modifies the U-Net structure. The network must be retrained. A classical self-attention block must be maintained as a parallel baseline.

---

## Section 8: Do Not Reduce Accuracy

A key constraint of clinical AI systems is that **experimental quantum methods must never compromise the performance of validated classical models**. In AURA, the DenseNet-121 vision model (macro-AUROC: 0.821) and the BraTS ResU-Net segmenter (composite Dice: 0.875) represent the clinical foundation.

To ensure quantum integration does not degrade accuracy, the following architectural rules must be enforced:

### Parallel Execution & Routing Guards
Quantum modules must run in parallel with their classical counterparts, never in series. The output of the classical model remains the primary clinical path unless the quantum model passes validation.

```
                  [ Preprocessed Study ]
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [ Classical Model ]               [ Quantum Model ]
     - DenseNet-121                    - PennyLane VQC
     - 3D ResU-Net                     - QKL Classifier
            │                                 │
            ▼                                 ▼
   [ Classical Posterior ]           [ Quantum Posterior ]
            │                                 │
            └───────────────┬─────────────────┘
                            ▼
                [ Wasserstein Conflict Guard ]
                            │
               Is distance > threshold (τ_t)?
                            ├──► Yes: Fallback to Classical + Flag
                            └──► No : Accept Quantum Decision
```

### Sandbox Calibration & Conformal Verification
All quantum outputs must pass through temperature scaling and Mondrian conformal prediction blocks. If the quantum posterior is uncalibrated, the conformal set size will widen, triggering an automated safety abstention (`LARGE_CONFORMAL_SET`).

### The Sign-Preserving Dual-Output Pattern
For interpretability tools (such as entanglement mapping or causal recommendation), classical attributions (Grad-CAM++, leave-one-out) must be displayed side-by-side with quantum metrics (connected correlators, measurement entropy). The system must label them clearly (e.g., "Model-Internal Quantum Correlation" vs. "Visual Pixel Attribution") so clinicians understand their differences.

---

## Section 9: System Gaps

This table identifies the existing clinical, engineering, security, and infrastructure gaps in the repository, along with proposed resolutions.

| Domain | Gap | Severity | Clinical / Technical Consequence | Proposed Resolution | Complexity |
|---|---|---|---|---|---|
| **Clinical** | No Brain Tumor Subtype Classification | **High** | The NeuroMind engine is limited to detecting "brain tumor" presence, leaving clinical differentials incomplete. | Integrate a QKL classifier trained on multi-class glioma, meningioma, and metastasis cohorts. | Medium |
| **Clinical** | No Longitudinal Growth Tracking | **Medium** | Clinicians cannot track tumor progression or treatment response over time. | Implement 3D spatial co-registration and volume difference tracking. | Medium |
| **Engineering** | In-Memory Memory Index | **Medium** | Case similarity features are lost whenever the FastAPI server restarts. | Replace the in-memory array with a local SQLite FTS5 database or Qdrant engine. | Low |
| **Engineering** | Simulated EHR Lab Integration | **Low** | Laboratory and symptom fields are simulated for testing because MIMIC-CXR lacks matched EHR records. | Map and ingest EHR cohorts from the MIMIC-IV clinical database. | Medium |
| **Security** | Windows App Control DLL Blocks | **High** | Compiled Python extensions (e.g., pandas) fail to load under restrictive Application Control policies. | Package the system in a secure Docker container or sign compiled DLLs. | Medium |
| **UI / UX** | Coarse Anatomical Overlay Boxes | **Medium** | The web dashboard renders static bounding box rectangles rather than dynamic contours. | Extract bounding contours directly from the Grad-CAM++ activation maps. | Medium |
| **Testing** | commented out Dev Dependencies | **Low** | Dev libraries (`pydicom`, `nibabel`, `pynrrd`) skip testing if not manually installed. | Move dev libraries into a structured `requirements-dev.txt` file. | Low |
| **Quantum** | Experimental Components Offline | **Medium** | The barren-plateau-aware encoders and re-uploading circuits are written but unused. | Train and calibrate the joint projection models, adding them as a selectable configuration. | Medium |

---

## Section 10: Hackathon Judge Review

If evaluated by a panel of medical AI, quantum computing, and clinical software judges, AURA would receive the following scores:

### Score Sheet (Out of 10)

- **Innovation: 9.0 / 10**
  *Justification*: The combination of conformal prediction, temperature scaling, and quantum VQC fusion is highly innovative. The concept of "calibrated doubt" sets the project apart from standard medical AI tools.
- **Technical Depth: 9.5 / 10**
  *Justification*: The codebase is clean and well-engineered. The presence of a 3D ResU-Net, a DenseNet-121, and a PennyLane VQC with simulated shot-noise propagation shows significant technical depth.
- **Quantum Innovation: 8.5 / 10**
  *Justification*: QMBA and connected correlator differential mapping are novel. However, the core quantum circuit remains a simulated 8-qubit model, and the more advanced concepts (data re-uploading, joint projections) are not yet integrated into the live pipeline.
- **Medical AI: 9.0 / 10**
  *Justification*: Preprocessing pipelines, Platt calibration, and Mondrian conformal prediction sets demonstrate a strong understanding of medical imaging requirements.
- **Clinical Utility: 9.0 / 10**
  *Justification*: Features like the sequential test recommender (EVOI) and clinician feedback loops are highly valuable. The utility is currently limited by the lack of brain tumor subtype classification and PACS integrations.
- **Architecture: 9.5 / 10**
  *Justification*: The async event bus, modular adapter contracts, and clean separation of concerns are well-implemented.
- **Scalability: 8.0 / 10**
  *Justification*: The system is designed for edge deployments. However, it relies on local SQLite databases and in-memory caches, which limits its ability to scale to multi-site hospital environments.
- **Novelty: 9.0 / 10**
  *Justification*: The integration of quantum measurement physics with clinical safety checks is unique.
- **Presentation: 8.5 / 10**
  *Justification*: The zero-dependency SPA dashboard is fast and functional. The presentation would be improved by rendering dynamic Grad-CAM contours instead of static anatomical bounding boxes.
- **Production Readiness: 8.5 / 10**
  *Justification*: The chest radiograph pipeline is production-ready. The brain MRI pipeline has a functional ResU-Net, but lacks subtype classification and longitudinal co-registration tools.

---

## Section 11: Roadmap

A prioritized roadmap to guide AURA from a hackathon prototype to a clinically validated, quantum-augmented medical OS.

```
       [ Immediate (1 Month) ]               [ Medium-Term (3-6 Months) ]              [ Long-Term (6-12 Months) ]
                  │                                        │                                        │
                  ▼                                        ▼                                        ▼
    - Grad-CAM++ Contour Drawing             - PACS Integration (C-STORE)             - 16-Qubit QBN Reasoner
    - SQLite Memory Persistence              - Brain Tumor Subtype Classifier         - Clinical Pilot Deployments
    - Dev Dependency Management              - Longitudinal Tracking Engine           - FDA 510(k) Pre-Submission
```

### Phase 1: Immediate Improvements (1 Month)
- **Grad-CAM++ Contour Drawing**: Modify [`apps/web/`](file:///e:/AURA/aura-main/aura/apps/web) and [`services/explain/engine.py`](file:///e:/AURA/aura-main/aura/services/explain/engine.py) to extract bounding contours from Grad-CAM++ maps, drawing precise regional overlays on the dashboard.
- **SQLite Memory Persistence**: Replace the in-memory cosine similarity cache in [`services/memory/engine.py`](file:///e:/AURA/aura-main/aura/services/memory/engine.py) with an SQLite FTS5 table, ensuring patient search histories persist across restarts.
- **Dependency Consolidation**: Move commented-out libraries (`pydicom`, `nibabel`, `pynrrd`) into a `requirements-dev.txt` file, ensuring the brain MRI test suite executes successfully on clean checkouts.

### Phase 2: Medium-Term Improvements (3–6 Months)
- **PACS Integration**: Implement a local DICOM listener (C-STORE SCP) to automatically receive uploads from hospital PACS systems.
- **Brain Tumor Subtype Classifier**: Integrate a QKL classifier to categorize brain tumors into Glioma, Meningioma, and Metastasis, expanding the clinical utility of the NeuroMind engine.
- **Longitudinal Tracking Engine**: Add spatial co-registration and volume difference calculations to track tumor growth or regression across time series.
- **Dockerization**: Containerize the gateway and database to bypass Windows Application Control policy blocks on native DLLs.

### Phase 3: Long-Term Improvements (6–12 Months)
- **16-Qubit QBN Reasoner**: Expand the VQC to 16 qubits, implementing a Quantum Bayesian Network to model joint dependencies between medical history, laboratory tests, and imaging findings.
- **Clinical Pilot Deployments**: Deploy AURA in shadow-mode at academic pilot sites to evaluate time-saving and diagnostic safety metrics.
- **FDA 510(k) Pre-Submission**: Compile validation metrics on multi-site clinical datasets to prepare for FDA 510(k) SaMD (Software as a Medical Device) clearance.

---

## Section 12: Final Recommendations

Twenty-five prioritized technical recommendations, ranked by their impact on clinical safety, innovation, and engineering readiness.

| Rank | Recommendation | Target Module | Difficulty | Expected Benefit | Hackathon Impact | Clinical Impact | Engineering Effort | Quantum Contribution | Priority |
|---|---|---|---|---|---|---|---|---|---|
| **1** | Extract bounding contours from Grad-CAM++ activations | `services/explain` | Medium | Replaces static boxes with precise visual grounding | High | High | Medium | None | **Critical** |
| **2** | Persist embedding similarity index in SQLite | `services/memory` | Low | Prevents case history loss on server restarts | Medium | Medium | Low | None | **High** |
| **3** | package application in Docker containers | Infrastructure | Medium | Bypasses Windows App Control DLL blocks | Low | High | Medium | None | **High** |
| **4** | Train and integrate Brain Tumor Subtype Classifier | `backend/engines/neuro`| High | Moves beyond binary presence to differential diagnosis | High | High | High | None | **High** |
| **5** | Implement 3D spatial co-registration for MRIs | `backend/foundation` | High | Enables longitudinal tracking of tumor volumes | Medium | High | High | None | **High** |
| **6** | Implement Quantum Bayesian Network for reasoning | `services/reasoning` | High | Replaces classical rules with quantum joint probabilities | High | Medium | High | High | **High** |
| **7** | Consolidate dev dependencies into dev requirements | Testing | Low | Prevents silent skipping of brain MRI tests | Low | Low | Low | None | **Medium** |
| **8** | Wire and calibrate JointProjection model | `services/fusion` | Medium | Compresses 1024-d embeddings to VQC angles | High | Medium | Medium | High | **Medium** |
| **9** | Implement PACS DICOM C-STORE SCP listener | Infrastructure | Medium | Automates ingestion from hospital networks | Low | High | Medium | None | **Medium** |
| **10** | Wire the Data Re-uploading QNode | `services/fusion` | Medium | Mitigates VQC barren plateaus in training | High | Low | Medium | High | **Medium** |
| **11** | Integrate QKL classifier for MRI subtype classification | `backend/engines/neuro`| High | Uses quantum kernels to improve rare class boundaries | High | High | High | High | **Medium** |
| **12** | Map and ingest real EHR tables from MIMIC-IV | `ml/training` | High | Replaces simulated lab results with real records | Low | High | High | None | **Medium** |
| **13** | Implement Quantum Attention in ResU-Net bottleneck | `backend/vision/brain` | High | Captures 3D spatial dependencies in MRI volumes | High | Medium | High | High | **Medium** |
| **14** | Add automated unit tests for ACI storage updates | Testing | Low | Prevents regressions in conformal feedback loops | Low | Low | Low | None | **Medium** |
| **15** | Add progress bar to the dashboard upload UI | UI/UX | Low | Improves user experience during large MRI uploads | Low | Low | Low | None | **Low** |
| **16** | Implement local QAOA patient workflow scheduler | Services | High | Optimizes clinical resource utilization | High | Low | High | High | **Low** |
| **17** | Add automated bias field checks to quality inspector | `backend/foundation` | Medium | Detects scanner artifacts before segmentation | Low | Medium | Medium | None | **Low** |
| **18** | Expand conformal prediction sets to class-conditional | `services/safety` | Medium | Ensures coverage guarantees for rare pathologies | Low | Medium | Medium | None | **Low** |
| **19** | Implement Quantum Federated Learning loops | Research | High | Enables secure multi-institutional training | High | Medium | High | High | **Low** |
| **20** | Run ablation studies comparing VQC and PoE ECE | Research | Medium | Validates quantum calibration improvements | High | Low | Medium | High | **Low** |
| **21** | Add PDF export option for clinical reports | `services/report` | Low | Enables easy sharing and archiving of results | Low | Medium | Low | None | **Low** |
| **22** | Integrate motion artifact detection in MRI quality | `backend/foundation` | Medium | Flags corrupted scans before processing | Low | Medium | Medium | None | **Low** |
| **23** | Add unit tests for the Wasserstein Conflict Guard | Testing | Low | Verifies safety loops under extreme distribution shifts | Low | Low | Low | None | **Low** |
| **24** | Implement Quantum Digital Twin representations | Research | High | Models patient states in Hilbert space | High | Low | High | High | **Low** |
| **25** | Add dark mode toggle to the dashboard UI | UI/UX | Low | Reduces eye strain for radiologists in dark rooms | Low | Low | Low | None | **Low** |
