"""Trainable classical bottleneck: joint (vision ⊕ clinical) vector -> n_qubits angles.

.. admonition:: STATUS — EXPERIMENTAL, NOT WIRED INTO THE SERVING PATH
   :class: warning

   This module is complete and correct but has **no importer in the running
   pipeline**: the shipped fusion path angle-encodes the hand-designed 8-channel
   ``evidence.encode`` vector directly (``services.fusion.quantum``), it does not
   route the 1024-d DenseNet embedding through this projection. It is retained as
   the designed extension point for learning the vision→qubit bottleneck jointly
   with the VQC (pair with ``device.make_reuploading_qnode``). Do not document it
   as active. See ``docs/ARCHITECTURE.md`` and audit §3.5 / §11.1.

Why this exists
---------------
``evidence.encode`` produces an 8-channel *hand-designed* evidence vector, one
channel per qubit. That is fine for the shipped demo, but the production vision
backbone (``services.vision.cnn.CXRBackbone``) emits a **1024-d DenseNet
embedding**. Angle-encoding a 1024-d vector needs either 1024 qubits (impossible
on the simulator, meaningless on near-term hardware) or a naive truncation that
throws away most of the signal.

Worse, feeding a high-dimensional, high-variance vector straight into a VQC is
the textbook trigger for a **barren plateau**: for an ``n``-qubit circuit whose
parameters look like a 2-design, the gradient variance decays as ``Var[∂C] ~
2**(-n)`` (McClean et al. 2018). You cannot fix that downstream — it has to be
fixed at the encoding boundary by (a) keeping ``n`` small and (b) presenting the
data at a scale where single-qubit rotations stay in their informative regime.

This module is that boundary. ``JointProjection`` is a trainable
``Linear(d_in -> n_qubits)`` followed by ``Tanh``, so the joint vector is
compressed to *exactly* ``n_qubits`` features in ``[-1, 1]`` before encoding. The
``Tanh`` bound is deliberate: the re-uploading QNode encodes each feature as an
angle ``π · x_i`` (see ``device.make_reuploading_qnode``), and bounding the input
to ``[-1, 1]`` keeps every encoded angle inside ``[-π, π]`` — one full,
non-aliased rotation, which is where the encoding gradient ``∂⟨Z⟩/∂x`` is largest.

Serving path is pure numpy (no torch dependency at inference); training holds the
same weights as a torch ``nn.Module`` so the projection is learned jointly with
the VQC by ordinary backprop.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common.config import ARTIFACTS

MODEL_VERSION = "fusion-projection-v1"


class JointProjection:
    """Linear + Tanh compression of a joint feature vector to ``n_qubits`` angles.

    Parameters are held as numpy arrays so serving needs no torch. ``to_torch``
    returns an equivalent ``nn.Module`` (built lazily) for joint VQC training.

      x_proj = tanh(W x + b),   W: (n_qubits, d_in),  b: (n_qubits,)

    The output lives in ``(-1, 1)`` and is fed to the re-uploading encoder.
    """

    def __init__(self, W: np.ndarray, b: np.ndarray):
        self.W = np.asarray(W, dtype=float)          # (n_qubits, d_in)
        self.b = np.asarray(b, dtype=float)          # (n_qubits,)
        self.n_qubits = self.W.shape[0]
        self.d_in = self.W.shape[1]
        self.model_version = MODEL_VERSION

    # ---- construction ----------------------------------------------------- #
    @classmethod
    def init_variance_preserving(cls, d_in: int, n_qubits: int, seed: int = 7) -> "JointProjection":
        """Xavier/Glorot init scaled so the pre-activation variance is ~1.

        Glorot keeps ``Var[Wx] ≈ Var[x]`` for a fan-in of ``d_in``; with the
        ``Tanh`` operating near its unit-slope region the encoded angles then have
        controlled spread. This is the encoding-side half of barren-plateau
        mitigation: the circuit-side half (local cost + shallow re-uploading) lives
        in ``device.make_reuploading_qnode``.
        """
        rng = np.random.default_rng(seed)
        limit = np.sqrt(6.0 / (d_in + n_qubits))     # Glorot uniform
        W = rng.uniform(-limit, limit, size=(n_qubits, d_in))
        b = np.zeros(n_qubits)
        return cls(W, b)

    @classmethod
    def load(cls, path: Path | None = None) -> "JointProjection | None":
        path = path or (ARTIFACTS / "fusion_projection.npz")
        if not Path(path).exists():
            return None
        d = np.load(path)
        return cls(d["W"], d["b"])

    def save(self, path: Path | None = None) -> None:
        path = path or (ARTIFACTS / "fusion_projection.npz")
        np.savez(path, W=self.W, b=self.b)

    # ---- serving (numpy) -------------------------------------------------- #
    def project(self, x: np.ndarray) -> np.ndarray:
        """Compress a joint vector (or batch) to ``n_qubits`` angles in (-1, 1)."""
        x = np.asarray(x, dtype=float)
        z = x @ self.W.T + self.b
        return np.tanh(z)

    __call__ = project

    # ---- training (torch) ------------------------------------------------- #
    def to_torch(self):
        """Return an ``nn.Module`` sharing these weights, for joint VQC training."""
        import torch
        import torch.nn as nn

        lin = nn.Linear(self.d_in, self.n_qubits)
        with torch.no_grad():
            lin.weight.copy_(torch.tensor(self.W, dtype=lin.weight.dtype))
            lin.bias.copy_(torch.tensor(self.b, dtype=lin.bias.dtype))

        class _Proj(nn.Module):
            def __init__(self, linear):
                super().__init__()
                self.linear = linear

            def forward(self, x):
                return torch.tanh(self.linear(x))

        return _Proj(lin)


def build_joint_vector(vision_embedding: np.ndarray, clinical_vector: np.ndarray) -> np.ndarray:
    """Concatenate a (possibly high-dim) vision embedding with the clinical vector.

    Kept trivial on purpose: the *only* place the two modalities are joined before
    the quantum boundary, so the concatenation order is documented in exactly one
    spot. Both parts are expected pre-normalized to a comparable scale.
    """
    v = np.asarray(vision_embedding, dtype=float).ravel()
    c = np.asarray(clinical_vector, dtype=float).ravel()
    return np.concatenate([v, c])
