"""Unit tests for the integrated quantum modules: QAE, QKL, QBN, and QMMF.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.engines.neuro.qkl import QKLClassifier
from schemas.clinical import Diagnosis
from services.fusion.multimodal import UnifiedFusionEngine
from services.fusion.qae import QuantumAutoencoder
from services.reasoning.qbn import QuantumBayesianNetwork


def test_qae_compression():
    # Construct random 1024-d feature vector
    x = np.random.randn(1024)
    qae = QuantumAutoencoder()
    compressed = qae.compress(x)
    assert compressed.shape == (8,)
    assert np.all(compressed >= 0.0)
    assert np.all(compressed <= 1.0)


def test_qkl_classification():
    # Construct random 128-d feature vector
    x = np.random.randn(128)
    qkl = QKLClassifier()
    probs = qkl.predict_subtype(x)
    assert "glioma" in probs
    assert "meningioma" in probs
    assert "metastasis" in probs
    assert pytest.approx(probs["glioma"], abs=1e-3) == 0.3333


def test_qbn_reasoning():
    qbn = QuantumBayesianNetwork()
    priors = {d: 1.0 / 6.0 for d in Diagnosis}
    # features: cardiac_evidence, infectious_evidence, malignancy_obstructive_evidence
    features = [1.0, 0.0, 0.0]
    adjusted = qbn.reason(priors, features)
    assert len(adjusted) == 6
    assert pytest.approx(adjusted.sum(), abs=1e-3) == 1.0


def test_qmmf_fusion():
    qmmf = UnifiedFusionEngine()
    cxr = np.random.rand(8)
    brain = np.random.randn(128)
    probs = qmmf.fuse_multimodal(cxr, brain)
    assert len(probs) == 12
    assert pytest.approx(probs.sum(), abs=1e-3) == 1.0
