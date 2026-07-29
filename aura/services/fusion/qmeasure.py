"""Quantum measurement observables: what the circuit reveals beyond its prediction.

The serving path reads eight single-qubit expectations ``<Z_i>`` and maps them to
diagnosis logits. That throws away almost everything the quantum state knows. This
module measures the rest of it, and turns it into two clinically meaningful signals
that a classical fusion model structurally cannot produce.

1. Evidence-entanglement map
----------------------------
The connected two-qubit correlator

    C_ij = <Z_i Z_j> - <Z_i><Z_j>

is exactly zero for every pair when the eight evidence channels are encoded on a
product state. It becomes non-zero only when the CNOT ring has entangled them. So
``C`` is a direct, per-patient readout of **which pieces of clinical evidence this
model has coupled** — effusion with cardiomegaly, opacity with consolidation, and so
on — measured from the quantum state rather than inferred by perturbing inputs.

That distinction is the point. Classical interaction attribution (Shapley
interactions, H-statistics, ablation) estimates coupling by *re-running the model on
altered inputs* and attributing the difference. ``C_ij`` is not an attribution: it is
a property of the state the circuit actually prepared for this patient, obtained in
the same measurement pass as the prediction. Nothing is perturbed and nothing is
approximated.

**The raw correlator is not attributable to the patient, and this is measured, not
assumed.** The trained ansatz rotations entangle the register regardless of the
input: on the shipped VQC, the *empty* evidence vector (no findings, no prior risk)
already carries total coupling 6.19, while a realistic effusion-plus-cardiomegaly
vector carries 2.29. Reporting the raw number as "this patient's evidence coupling"
would therefore be backwards — the emptiest input looks the most coupled.

So the reportable quantity is the **differential** ``C_ij(x) - C_ij(x_ref)`` against
a reference state, and the reference is the clinically meaningful one: the evidence
vector with every finding absent. That isolates what *this patient's findings* did to
the coupling structure from what the trained circuit does to any input. Both are kept
on the result — the raw map is the honest state property, the differential is the
honest attribution — and they are never conflated.

What neither of them is: a causal claim. A strong effusion-cardiomegaly correlator
says this VQC has learned to treat those two findings jointly; it does not say one
causes the other. The report wording preserves that distinction.

2. Measurement entropy
----------------------
Shannon entropy of the full 2^8 measurement distribution, in bits. Bounded by
``n_qubits`` (8 bits here). A near-product, near-deterministic state carries low
entropy; a state spread over many basis outcomes carries high entropy and is the
circuit saying the evidence configuration is not resolving to one pattern.

Both quantities come from the *same* trained circuit that produces the diagnosis. No
extra model, no extra training, no surrogate.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from .evidence import EVIDENCE_CHANNELS

#: Correlator magnitudes below this are numerical noise from the simulator, not
#: entanglement. float64 expectation values carry ~1e-16 error and the products
#: amplify it slightly; 1e-9 is far above that and far below any real coupling.
CORRELATION_FLOOR = 1e-9


@lru_cache(maxsize=8)
def _pairwise_qnode(n_qubits: int, n_layers: int, device_name: str = "default.qubit",
                    entangler: str = "ring"):
    """QNode returning every single-qubit ``<Z_i>`` and every pair ``<Z_i Z_j>``.

    One circuit, one pass, all observables. Measuring the pairs in a separate
    execution would double the simulator cost and — on hardware, where ``Z_i`` and
    ``Z_i Z_j`` are commuting observables measurable in the same basis — would waste
    half the shot budget for no reason.

    Cached because building a QNode compiles the tape; a per-request rebuild would
    dominate the inference time of an 8-qubit circuit.
    """
    import pennylane as qml

    dev = qml.device(device_name, wires=n_qubits, shots=None)
    pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]

    @qml.qnode(dev, interface="numpy", diff_method=None)
    def circuit(x, theta):
        # Identical preparation to ``device.make_qnode`` — same encoding, same
        # ansatz. It has to be: this measures the state that produced the diagnosis,
        # not a state that merely resembles it.
        for i in range(n_qubits):
            qml.RY(np.pi * x[..., i], wires=i)
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(theta[layer][i][0], wires=i)
                qml.RZ(theta[layer][i][1], wires=i)
            if entangler == "ring":
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
        singles = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        doubles = [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for i, j in pairs]
        return singles + doubles

    return circuit, pairs


@dataclass(frozen=True)
class EvidenceEntanglement:
    """Per-patient map of which evidence channels the circuit coupled."""

    #: Connected correlators of the state that produced this diagnosis,
    #: ``(n_qubits, n_qubits)``, symmetric, zero diagonal. A property of the state —
    #: it includes the trained ansatz's data-independent entanglement.
    correlation: np.ndarray
    #: The same correlators for the reference state (all findings absent).
    baseline_correlation: np.ndarray
    #: Single-qubit expectations ``<Z_i>``, in ``[-1, 1]``.
    expectations: np.ndarray
    #: Channel names, aligned with the matrix axes.
    channels: tuple[str, ...]
    #: Shannon entropy of the 2^n measurement distribution, in bits.
    measurement_entropy_bits: float
    #: Entropy of the reference state, for the same reason the baseline map exists.
    baseline_entropy_bits: float
    #: Maximum entropy this register could carry — ``n_qubits`` bits. Given so a
    #: reader can normalise without knowing the circuit width.
    max_entropy_bits: float

    @property
    def differential(self) -> np.ndarray:
        """``C(x) - C(x_ref)``: what this patient's findings did to the coupling.

        The attributable quantity. Positive entries mean the patient's evidence
        *strengthened* the coupling between those two channels relative to a study
        with no findings; negative entries mean it broke coupling the circuit
        otherwise carries.
        """
        return self.correlation - self.baseline_correlation

    @property
    def total_coupling(self) -> float:
        """Sum of |C_ij| over all pairs of the served state.

        Note this is **not** zero for an empty evidence vector — the trained
        rotations entangle the register on their own. Use :attr:`differential_coupling`
        for the patient-attributable number.
        """
        return float(np.abs(np.triu(self.correlation, k=1)).sum())

    @property
    def differential_coupling(self) -> float:
        """Sum of |C_ij(x) - C_ij(x_ref)| — how much this patient's evidence moved
        the coupling structure. This is the number to report per patient."""
        return float(np.abs(np.triu(self.differential, k=1)).sum())

    @property
    def baseline_coupling(self) -> float:
        """Coupling the circuit carries with no evidence at all. A model constant."""
        return float(np.abs(np.triu(self.baseline_correlation, k=1)).sum())

    @property
    def entropy_shift_bits(self) -> float:
        """Change in measurement entropy caused by this patient's evidence.

        Negative means the findings *concentrated* the measurement distribution — the
        circuit resolved toward fewer outcomes than it does on an empty study, which
        is the behaviour informative evidence should produce.
        """
        return float(self.measurement_entropy_bits - self.baseline_entropy_bits)

    @property
    def normalised_entropy(self) -> float:
        """Measurement entropy as a fraction of the register's capacity, in [0, 1]."""
        return float(self.measurement_entropy_bits / self.max_entropy_bits)

    def top_pairs(self, k: int = 5, *, differential: bool = True
                  ) -> list[dict[str, Any]]:
        """The ``k`` most strongly coupled evidence pairs, strongest first.

        Ranks by the differential by default, because that is the patient-attributable
        quantity; pass ``differential=False`` to rank the raw state correlators.

        Sign is preserved and it is meaningful. A positive value means the two
        channels' ``Z`` readouts move together; negative means they oppose. Reporting
        only magnitude would discard half the finding.
        """
        matrix = self.differential if differential else self.correlation
        pairs: list[dict[str, Any]] = []
        n = matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                value = float(matrix[i, j])
                if abs(value) < CORRELATION_FLOOR:
                    continue
                pairs.append({
                    "channels": (self.channels[i], self.channels[j]),
                    "correlation": round(value, 6),
                    "raw_correlation": round(float(self.correlation[i, j]), 6),
                    "direction": "aligned" if value > 0 else "opposed",
                })
        pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
        return pairs[:k]

    def is_product_state(self) -> bool:
        """True when the served state carries no pairwise coupling at all.

        Rare on a trained circuit — the ansatz rotations entangle the register
        independently of the input — so this is mostly a guard for an untrained or
        zeroed ``theta``. It is checked rather than assumed because an untrained VQC
        silently behaving as a product state would make every coupling claim vacuous.
        """
        return self.total_coupling < CORRELATION_FLOOR

    def to_dict(self) -> dict[str, Any]:
        return {
            "channels": list(self.channels),
            "correlation": [[round(float(v), 6) for v in row]
                            for row in self.correlation],
            "differential": [[round(float(v), 6) for v in row]
                             for row in self.differential],
            "expectations": [round(float(v), 6) for v in self.expectations],
            "total_coupling": round(self.total_coupling, 6),
            "baseline_coupling": round(self.baseline_coupling, 6),
            "differential_coupling": round(self.differential_coupling, 6),
            "measurement_entropy_bits": round(self.measurement_entropy_bits, 4),
            "baseline_entropy_bits": round(self.baseline_entropy_bits, 4),
            "entropy_shift_bits": round(self.entropy_shift_bits, 4),
            "max_entropy_bits": round(self.max_entropy_bits, 4),
            "normalised_entropy": round(self.normalised_entropy, 4),
            "product_state": self.is_product_state(),
            "top_pairs": self.top_pairs(),
            "interpretation": (
                "C_ij = <Z_i Z_j> - <Z_i><Z_j>, measured on the state that produced "
                "this diagnosis. 'correlation' is the raw state property and includes "
                "the trained ansatz's data-independent entanglement; 'differential' "
                "subtracts the all-findings-absent reference and is the quantity "
                "attributable to this patient's evidence. Both describe coupling "
                "within the model, not causation in the patient."
            ),
        }


def _correlators(model: Any, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Connected two-qubit correlators and single-qubit expectations for one input."""
    circuit, pairs = _pairwise_qnode(model.n_qubits, model.n_layers,
                                     entangler=getattr(model, "entangler", "ring"))
    values = np.asarray([float(v) for v in circuit(x, model.theta)], dtype=float)
    singles = values[:model.n_qubits]
    doubles = values[model.n_qubits:]

    correlation = np.zeros((model.n_qubits, model.n_qubits), dtype=float)
    for (i, j), joint in zip(pairs, doubles):
        connected = float(joint - singles[i] * singles[j])
        if abs(connected) < CORRELATION_FLOOR:
            connected = 0.0
        correlation[i, j] = correlation[j, i] = connected
    return correlation, singles


def measure_entanglement(model: Any, x: np.ndarray,
                         reference: np.ndarray | None = None) -> EvidenceEntanglement:
    """Measure the evidence-entanglement map for one evidence vector.

    Args:
        model: a trained :class:`~aura.services.fusion.quantum.QuantumFusion`.
        x: the 8-channel evidence vector this patient produced.
        reference: the baseline to attribute against. Defaults to the zero vector —
            every finding absent, no prior risk — which is the clinically meaningful
            "nothing found" state and makes the differential read as "what these
            findings contributed".

    Raises:
        ValueError: ``x`` does not match the circuit width. Silently truncating or
            padding would produce a plausible map of the wrong patient.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size != model.n_qubits:
        raise ValueError(
            f"evidence vector has {x.size} channels but the circuit has "
            f"{model.n_qubits} qubits")

    reference_vector = (np.zeros(model.n_qubits) if reference is None
                        else np.asarray(reference, dtype=float).ravel())
    correlation, singles = _correlators(model, x)
    baseline, _ = _baseline_for(model, reference_vector)

    channels = tuple(EVIDENCE_CHANNELS[:model.n_qubits])
    return EvidenceEntanglement(
        correlation=correlation,
        baseline_correlation=baseline,
        expectations=singles,
        channels=channels,
        measurement_entropy_bits=model.measurement_entropy(x),
        baseline_entropy_bits=_baseline_for(model, reference_vector)[1],
        max_entropy_bits=float(model.n_qubits),
    )


#: Reference-state cache, keyed by circuit parameters. A model constant computed
#: once per (theta, reference) pair rather than once per request.
_BASELINE_CACHE: dict[tuple, tuple[np.ndarray, float]] = {}


def _baseline_for(model: Any, reference: np.ndarray) -> tuple[np.ndarray, float]:
    key = (id(model), model.theta.tobytes(), reference.tobytes())
    cached = _BASELINE_CACHE.get(key)
    if cached is None:
        correlation, _ = _correlators(model, reference)
        entropy = model.measurement_entropy(reference)
        cached = (correlation, entropy)
        _BASELINE_CACHE[key] = cached
    return cached


def coupling_summary(entanglement: EvidenceEntanglement, top_k: int = 3) -> str:
    """One clinician-readable sentence about what the circuit coupled.

    Deliberately hedged. The correlator is a model property, and a report sentence
    that reads as a claim about the patient's physiology would be exactly the kind of
    ungrounded statement the rest of AURA's report engine exists to prevent.
    """
    if entanglement.is_product_state():
        return ("The quantum fusion state carries no evidence coupling for this "
                "study: the findings were assessed independently.")
    pairs = entanglement.top_pairs(top_k)
    if not pairs:
        return ("This study's evidence did not measurably change the fusion state's "
                "coupling structure relative to a study with no findings.")
    described = "; ".join(
        f"{p['channels'][0]} and {p['channels'][1]} ({p['direction']}, "
        f"{abs(p['correlation']):.3f})" for p in pairs)
    return (f"Relative to a study with no findings, this evidence most changed the "
            f"quantum fusion state's coupling for: {described}. This reflects how "
            f"the model weighed these findings together, not a causal relationship "
            f"in the patient.")


__all__ = ["EvidenceEntanglement", "measure_entanglement", "coupling_summary",
           "CORRELATION_FLOOR"]
