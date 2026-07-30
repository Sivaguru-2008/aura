"""Quantum Kernel Learning head: kernel correctness, calibration, and honesty.

The honesty tests matter as much as the numerical ones. The classifier is trained
on BraTS glioma grade (HGG/LGG) because that is the only tumour label axis AURA
holds; it must never emit a meningioma or metastasis probability, and it must
degrade to an abstention — not a crash, and not a confident guess — when its
weights are missing.
"""
from __future__ import annotations

import numpy as np
import pytest

from aura.backend.engines.neuro.qkl import (
    DEFAULT_WEIGHTS,
    LEGACY_SUBTYPES,
    QKLClassifier,
)

TRAINED = pytest.mark.skipif(
    not DEFAULT_WEIGHTS.exists(),
    reason="qkl_classifier.npz not built (run: python -m aura.aura_cli qkl --train)",
)


# --------------------------------------------------------------------------- #
# Quantum kernel
# --------------------------------------------------------------------------- #
def test_kernel_is_a_valid_fidelity_kernel():
    """K(x,x)=1, symmetry, and bounds in [0,1] — the defining properties."""
    clf = QKLClassifier()
    rng = np.random.default_rng(0)
    a, b = rng.random(clf.n_qubits), rng.random(clf.n_qubits)

    assert clf.kernel_eval(a, a) == pytest.approx(1.0, abs=1e-9)
    assert clf.kernel_eval(a, b) == pytest.approx(clf.kernel_eval(b, a), abs=1e-9)
    assert 0.0 <= clf.kernel_eval(a, b) <= 1.0


def test_statevector_and_adjoint_circuit_agree():
    """The two evaluation modes are the same mathematics, so they must match.

    ``statevector`` builds the Gram matrix for training; ``sampled`` is the
    adjoint circuit that executes on hardware. A divergence here would mean the
    trained model and the deployed circuit compute different kernels.
    """
    clf = QKLClassifier()
    rng = np.random.default_rng(1)
    for _ in range(5):
        a, b = rng.random(clf.n_qubits), rng.random(clf.n_qubits)
        assert clf.kernel_eval(a, b, mode="statevector") == pytest.approx(
            clf.kernel_eval(a, b, mode="sampled"), abs=1e-9
        )


def test_gram_matrix_is_psd():
    """A kernel SVM is only well-posed on a positive-semidefinite Gram matrix."""
    clf = QKLClassifier()
    rng = np.random.default_rng(2)
    K = clf.gram(rng.random((12, clf.n_qubits)))

    assert K.shape == (12, 12)
    assert np.allclose(K, K.T, atol=1e-9)
    assert np.diag(K) == pytest.approx(np.ones(12), abs=1e-9)
    assert np.linalg.eigvalsh(K).min() > -1e-8


# --------------------------------------------------------------------------- #
# Untrained fallback — the "weights unavailable" contract
# --------------------------------------------------------------------------- #
def test_untrained_classifier_abstains_rather_than_guessing():
    pred = QKLClassifier().predict(np.random.randn(128))

    assert pred.trained is False
    assert pred.abstained is True
    assert pred.top_label == "undetermined"
    assert pred.abstain_reason and "no trained weights" in pred.abstain_reason
    assert set(pred.untrained_labels) == set(LEGACY_SUBTYPES)


def test_untrained_legacy_surface_returns_uniform_prior():
    """Historical contract: predict_subtype() is a uniform prior when untrained."""
    probs = QKLClassifier().predict_subtype(np.random.randn(128))

    assert set(probs) == set(LEGACY_SUBTYPES)
    assert probs["glioma"] == pytest.approx(0.3333, abs=1e-3)


def test_load_of_missing_weights_never_raises(tmp_path):
    clf = QKLClassifier.load(tmp_path / "does_not_exist.npz")
    assert clf.is_trained is False


def test_load_of_corrupt_weights_falls_back(tmp_path):
    """A truncated artefact must degrade to untrained, not take the engine down."""
    bad = tmp_path / "corrupt.npz"
    bad.write_bytes(b"PK\x03\x04 not really an npz")

    clf = QKLClassifier.load(bad)
    assert clf.is_trained is False


def test_wrong_embedding_width_is_rejected():
    with pytest.raises(ValueError, match="128-d embedding"):
        QKLClassifier()._project(np.zeros(64))


# --------------------------------------------------------------------------- #
# Trained model
# --------------------------------------------------------------------------- #
@TRAINED
def test_trained_model_loads_with_expected_shape():
    clf = QKLClassifier.load()

    assert clf.is_trained
    assert clf.task == "glioma_grade"
    assert clf.classes == ("HGG", "LGG")
    assert clf.support_vectors.shape[1] == clf.n_qubits
    assert clf.alpha.shape[0] == clf.support_vectors.shape[0]


@TRAINED
def test_trained_prediction_is_a_normalised_distribution():
    clf = QKLClassifier.load()
    pred = clf.predict(np.random.randn(128))

    assert pred.trained is True
    assert pred.abstained is False
    assert sum(pred.labels.values()) == pytest.approx(1.0, abs=1e-3)
    assert pred.top_label in clf.classes
    assert 0.0 <= pred.confidence <= 1.0
    assert 0.0 <= pred.kernel_similarity["mean"] <= 1.0
    assert pred.kernel_similarity["support_vectors"] == len(clf.support_vectors)


@TRAINED
def test_trained_model_never_asserts_an_untrained_subtype():
    """The core anti-fabrication guarantee.

    The head is trained on glioma grade. Even trained, the legacy three-class
    surface must stay a uniform prior, because AURA has no meningioma or
    metastasis imaging to have learned from.
    """
    clf = QKLClassifier.load()
    probs = clf.predict_subtype(np.random.randn(128))

    assert all(v == pytest.approx(0.3333, abs=1e-3) for v in probs.values())
    assert "meningioma" not in clf.classes
    assert "metastasis" not in clf.classes


@TRAINED
def test_prediction_is_deterministic():
    clf = QKLClassifier.load()
    x = np.random.default_rng(3).standard_normal(128)
    assert clf.predict(x).labels == clf.predict(x).labels


@TRAINED
def test_round_trip_save_load_preserves_predictions(tmp_path):
    clf = QKLClassifier.load()
    x = np.random.default_rng(4).standard_normal(128)
    before = clf.predict(x)

    path = clf.save(tmp_path / "rt.npz")
    after = QKLClassifier.load(path).predict(x)

    assert after.labels == before.labels
    assert after.task == before.task
    assert after.provenance == before.provenance


@TRAINED
def test_held_out_performance_is_better_than_chance():
    """Guards against a silently degenerate retrain shipping as 'trained'."""
    import json

    from aura.ml.training.train_qkl import REPORT_PATH

    if not REPORT_PATH.exists():
        pytest.skip("training report absent")
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["split"]["level"].startswith("subject"), "split must be subject-grouped"
    assert report["test_quantum"]["auroc"] > 0.55
    assert report["test_quantum"]["ece"] < 0.15


# --------------------------------------------------------------------------- #
# Report rendering
# --------------------------------------------------------------------------- #
def test_report_text_abstains_when_head_is_untrained():
    from aura.backend.engines.neuro.bundle import _differential_text

    text = _differential_text(QKLClassifier().predict(np.random.randn(128)).to_dict())

    assert "no subtype is being asserted" in text
    assert "abstained" in text
    assert "%" not in text.split("abstained")[0] or "33.3%" not in text


def test_report_text_without_any_qkl_payload():
    from aura.backend.engines.neuro.bundle import _differential_text

    assert "no subtype is being asserted" in _differential_text(None)


@TRAINED
def test_report_text_quotes_the_trained_task_and_its_uncertainty():
    from aura.backend.engines.neuro.bundle import _differential_text

    text = _differential_text(QKLClassifier().load().predict(np.random.randn(128)).to_dict())

    assert "glioma_grade" in text
    assert "HGG" in text and "LGG" in text
    assert "meningioma" not in text.lower().split("never asserted")[0]
    assert "not a validated diagnostic claim" in text
