"""Quantum Kernel Learning (QKL) / QSVM over ResU-Net brain embeddings.

Projects the 128-dimensional encoder bottleneck into a 6-qubit Hilbert space with
an IQP feature map and classifies with a kernel SVM whose Gram matrix is the
quantum *fidelity kernel*

    K(x, x') = |<phi(x) | phi(x')>|^2

which is exactly the ``probs[0]`` readout of the adjoint encode/un-encode circuit.

Two evaluation modes, same mathematics:

``statevector``
    Encode once per sample, then ``K = |Phi Phi^dagger|^2`` as a single matmul.
    Exact, and the only tractable way to build an N x N training Gram matrix.
``sampled``
    Runs the adjoint circuit with finite shots — the form that executes on real
    hardware (see :mod:`aura.services.fusion.device`). Noisy, and what the
    ``AURA_USE_REAL_QPU`` path uses.

Label honesty
-------------
The shipped weights are trained on the **only** brain-tumour label axis present in
AURA's corpus: BraTS-2020 glioma grade (HGG vs LGG). AURA has no meningioma or
metastasis imaging, so those classes are *not* trained and are never asserted —
:meth:`QKLClassifier.predict` reports them as untrained and abstains. An untrained
classifier (no weights on disk) keeps the historical uniform-prior behaviour so
callers that predate training are unaffected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pennylane as qml

from aura.common.config import ARTIFACTS

#: Legacy three-class surface kept for backward compatibility with callers that
#: predate training (``bundle.py``, ``discussion.py``, the neuro engine metadata).
LEGACY_SUBTYPES = ("glioma", "meningioma", "metastasis")

#: Default artefact location.
DEFAULT_WEIGHTS = ARTIFACTS / "brain" / "qkl_classifier.npz"


@dataclass(frozen=True)
class QKLPrediction:
    """A single QKL inference, with everything a report needs to ground it."""

    labels: dict[str, float]
    top_label: str
    confidence: float
    margin: float
    decision_value: float
    kernel_similarity: dict[str, Any]
    trained: bool
    task: str
    abstained: bool = False
    abstain_reason: str | None = None
    untrained_labels: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": dict(self.labels),
            "top_label": self.top_label,
            "confidence": round(float(self.confidence), 4),
            "margin": round(float(self.margin), 4),
            "decision_value": round(float(self.decision_value), 4),
            "kernel_similarity": self.kernel_similarity,
            "trained": bool(self.trained),
            "task": self.task,
            "abstained": bool(self.abstained),
            "abstain_reason": self.abstain_reason,
            "untrained_labels": list(self.untrained_labels),
            "provenance": dict(self.provenance),
        }


class QKLClassifier:
    """Quantum-kernel SVM with Platt-scaled probabilities.

    Constructed with no arguments the classifier is *untrained*: it holds no
    support vectors and returns a uniform prior, which is the documented
    fallback whenever ``artifacts/brain/qkl_classifier.npz`` is absent or
    unreadable. :meth:`load` never raises for a missing/corrupt artefact.
    """

    def __init__(
        self,
        W_proj: np.ndarray | None = None,
        b_proj: np.ndarray | None = None,
        alpha: np.ndarray | None = None,
        b_svm: float = 0.0,
        support_vectors: np.ndarray | None = None,
        support_labels: np.ndarray | None = None,
        n_qubits: int = 6,
        n_layers: int = 2,
        *,
        classes: Sequence[str] = LEGACY_SUBTYPES,
        platt_a: float = -1.0,
        platt_b: float = 0.0,
        task: str = "untrained",
        feature_mean: np.ndarray | None = None,
        feature_scale: np.ndarray | None = None,
        provenance: dict[str, Any] | None = None,
    ):
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.classes = tuple(str(c) for c in classes)
        self.task = str(task)
        self.platt_a = float(platt_a)
        self.platt_b = float(platt_b)
        self.provenance = dict(provenance or {})

        # Projection from the 128-d bottleneck down to n_qubits angles.
        self.W_proj = (
            np.zeros((self.n_qubits, 128), dtype=float)
            if W_proj is None
            else np.asarray(W_proj, dtype=float)
        )
        self.b_proj = (
            np.zeros(self.n_qubits, dtype=float)
            if b_proj is None
            else np.asarray(b_proj, dtype=float)
        )
        # Optional input standardisation fitted on the training split.
        self.feature_mean = (
            np.zeros(self.W_proj.shape[1], dtype=float)
            if feature_mean is None
            else np.asarray(feature_mean, dtype=float)
        )
        self.feature_scale = (
            np.ones(self.W_proj.shape[1], dtype=float)
            if feature_scale is None
            else np.asarray(feature_scale, dtype=float)
        )

        self.alpha = np.asarray(alpha, dtype=float) if alpha is not None else np.zeros(0)
        self.b_svm = float(b_svm)
        self.support_vectors = (
            np.asarray(support_vectors, dtype=float)
            if support_vectors is not None
            else np.zeros((0, self.n_qubits), dtype=float)
        )
        self.support_labels = (
            np.asarray(support_labels, dtype=float)
            if support_labels is not None
            else np.zeros(0)
        )

        self._dev = qml.device("default.qubit", wires=self.n_qubits)
        self._build_circuits()
        self._sv_states: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Quantum feature map
    # ------------------------------------------------------------------ #
    def _build_circuits(self) -> None:
        n = self.n_qubits

        def _encode(x):
            for i in range(n):
                qml.Hadamard(wires=i)
                qml.RZ(np.pi * x[i], wires=i)
            for i in range(n):
                j = (i + 1) % n
                qml.CNOT(wires=[i, j])
                qml.RZ(np.pi * x[i] * x[j], wires=j)
                qml.CNOT(wires=[i, j])

        @qml.qnode(self._dev, interface="numpy")
        def state_circuit(x):
            _encode(x)
            return qml.state()

        @qml.qnode(self._dev, interface="numpy")
        def iqp_kernel_circuit(x1, x2):
            _encode(x1)
            # Adjoint of the encoding for x2.
            for i in reversed(range(n)):
                j = (i + 1) % n
                qml.CNOT(wires=[i, j])
                qml.RZ(-np.pi * x2[i] * x2[j], wires=j)
                qml.CNOT(wires=[i, j])
            for i in range(n):
                qml.RZ(-np.pi * x2[i], wires=i)
                qml.Hadamard(wires=i)
            return qml.probs(wires=range(n))

        self._state_circuit = state_circuit
        self._kernel_circuit = iqp_kernel_circuit

    def feature_states(self, X: np.ndarray) -> np.ndarray:
        """Statevectors ``|phi(x)>`` for a batch of *already projected* angles."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.asarray([np.asarray(self._state_circuit(row)) for row in X])

    def gram(self, A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
        """Exact fidelity Gram matrix between projected angle batches."""
        SA = self.feature_states(A)
        SB = SA if B is None else self.feature_states(B)
        return np.abs(SA @ SB.conj().T) ** 2

    # ------------------------------------------------------------------ #
    # Projection
    # ------------------------------------------------------------------ #
    def _project(self, embedding: np.ndarray) -> np.ndarray:
        """128-d embedding -> n_qubits angles in [0, 1]."""
        e = np.asarray(embedding, dtype=float).ravel()
        if e.shape[0] != self.W_proj.shape[1]:
            raise ValueError(
                f"QKL expects a {self.W_proj.shape[1]}-d embedding, got {e.shape[0]}"
            )
        scale = np.where(np.abs(self.feature_scale) < 1e-12, 1.0, self.feature_scale)
        z = self.W_proj @ ((e - self.feature_mean) / scale) + self.b_proj
        return (np.tanh(z) + 1.0) / 2.0

    def project_batch(self, embeddings: np.ndarray) -> np.ndarray:
        return np.asarray([self._project(e) for e in np.atleast_2d(embeddings)])

    # ------------------------------------------------------------------ #
    # Kernel evaluation
    # ------------------------------------------------------------------ #
    def kernel_eval(self, x1: np.ndarray, x2: np.ndarray, mode: str = "statevector") -> float:
        """Fidelity kernel between two projected angle vectors.

        ``statevector`` is exact; ``sampled`` runs the adjoint circuit, which is
        the form that executes on real QPUs.
        """
        if mode == "sampled":
            probs = self._kernel_circuit(np.asarray(x1, float), np.asarray(x2, float))
            return float(probs[0])
        s1 = np.asarray(self._state_circuit(np.asarray(x1, float)))
        s2 = np.asarray(self._state_circuit(np.asarray(x2, float)))
        return float(np.abs(np.vdot(s2, s1)) ** 2)

    @property
    def is_trained(self) -> bool:
        return self.support_vectors.shape[0] > 0 and self.alpha.size > 0

    def _support_states(self) -> np.ndarray:
        if self._sv_states is None:
            self._sv_states = self.feature_states(self.support_vectors)
        return self._sv_states

    def decision_function(self, embedding: np.ndarray) -> tuple[float, np.ndarray]:
        """SVM decision value plus the kernel row against the support set."""
        x = self._project(embedding)
        sv_states = self._support_states()
        qx = np.asarray(self._state_circuit(x))
        k_row = np.abs(sv_states @ qx.conj()) ** 2
        return float(np.dot(self.alpha, k_row) + self.b_svm), k_row

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def predict(self, embedding: np.ndarray) -> QKLPrediction:
        """Full prediction with calibrated probabilities and kernel evidence."""
        if not self.is_trained:
            uniform = 1.0 / max(len(self.classes), 1)
            return QKLPrediction(
                labels={c: round(uniform, 4) for c in self.classes},
                top_label="undetermined",
                confidence=uniform,
                margin=0.0,
                decision_value=0.0,
                kernel_similarity={"support_vectors": 0, "mode": "untrained"},
                trained=False,
                task=self.task,
                abstained=True,
                abstain_reason=(
                    "QKL classifier has no trained weights "
                    f"({DEFAULT_WEIGHTS.name} absent); no subtype is asserted."
                ),
                untrained_labels=self.classes,
                provenance=dict(self.provenance),
            )

        decision, k_row = self.decision_function(embedding)
        # Platt scaling fitted on a held-out split: p(positive) = sigma(A*f + B).
        p_pos = 1.0 / (1.0 + np.exp(self.platt_a * decision + self.platt_b))
        p_pos = float(np.clip(p_pos, 1e-6, 1 - 1e-6))

        if len(self.classes) == 2:
            probs = {self.classes[0]: 1.0 - p_pos, self.classes[1]: p_pos}
        else:  # one-vs-rest scores already softmaxed by the trainer
            from aura.common.mathx import softmax

            vals = softmax(np.asarray([decision] * len(self.classes)))
            probs = {c: float(v) for c, v in zip(self.classes, vals)}

        ordered = sorted(probs.items(), key=lambda kv: -kv[1])
        top_label, top_p = ordered[0]
        runner_p = ordered[1][1] if len(ordered) > 1 else 0.0

        pos_mask = self.support_labels > 0
        similarity = {
            "mode": "fidelity_statevector",
            "support_vectors": int(self.support_vectors.shape[0]),
            "mean": round(float(k_row.mean()), 4),
            "max": round(float(k_row.max()), 4),
            "nearest_support_index": int(np.argmax(k_row)),
            "nearest_support_label": self.classes[int(self.support_labels[int(np.argmax(k_row))] > 0)]
            if len(self.classes) == 2
            else None,
            "mean_by_class": {
                self.classes[1]: round(float(k_row[pos_mask].mean()), 4) if pos_mask.any() else None,
                self.classes[0]: round(float(k_row[~pos_mask].mean()), 4) if (~pos_mask).any() else None,
            }
            if len(self.classes) == 2
            else {},
        }

        return QKLPrediction(
            labels={c: round(float(p), 4) for c, p in probs.items()},
            top_label=top_label,
            confidence=float(top_p),
            margin=float(top_p - runner_p),
            decision_value=float(decision),
            kernel_similarity=similarity,
            trained=True,
            task=self.task,
            provenance=dict(self.provenance),
        )

    def predict_subtype(self, embedding: np.ndarray) -> dict[str, float]:
        """Backward-compatible three-class surface.

        Untrained (or trained on a task that is not the three-class subtype
        problem) this returns the historical uniform prior, so no meningioma or
        metastasis probability is ever fabricated.
        """
        if not self.is_trained or set(self.classes) != set(LEGACY_SUBTYPES):
            return {c: 0.3333 for c in LEGACY_SUBTYPES}
        pred = self.predict(embedding)
        return {c: float(pred.labels.get(c, 0.0)) for c in LEGACY_SUBTYPES}

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Path | None = None) -> Path:
        import json

        path = Path(path or DEFAULT_WEIGHTS)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            W_proj=self.W_proj,
            b_proj=self.b_proj,
            alpha=self.alpha,
            b_svm=self.b_svm,
            support_vectors=self.support_vectors,
            support_labels=self.support_labels,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            classes=np.asarray(self.classes, dtype=object),
            platt_a=self.platt_a,
            platt_b=self.platt_b,
            task=self.task,
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            provenance=json.dumps(self.provenance),
        )
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "QKLClassifier":
        """Load trained weights; fall back to an untrained instance on any error."""
        import json

        path = Path(path or DEFAULT_WEIGHTS)
        if not path.exists():
            return cls()
        try:
            d = np.load(path, allow_pickle=True)

            def _get(key, default):
                return d[key] if key in d.files else default

            classes = _get("classes", None)
            classes = tuple(str(c) for c in classes) if classes is not None else LEGACY_SUBTYPES
            prov_raw = _get("provenance", None)
            provenance = json.loads(str(prov_raw)) if prov_raw is not None else {}
            return cls(
                d["W_proj"],
                d["b_proj"],
                d["alpha"],
                float(d["b_svm"]),
                d["support_vectors"],
                d["support_labels"],
                int(d["n_qubits"]),
                int(d["n_layers"]),
                classes=classes,
                platt_a=float(_get("platt_a", -1.0)),
                platt_b=float(_get("platt_b", 0.0)),
                task=str(_get("task", "unknown")),
                feature_mean=_get("feature_mean", None),
                feature_scale=_get("feature_scale", None),
                provenance=provenance,
            )
        except Exception:
            return cls()
