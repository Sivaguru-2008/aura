from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Any

@dataclass
class MetricsCard:
    backend: str
    accuracy: float = 0.0
    f1_macro: float = 0.0
    ece: float = 0.0
    n_eval: int = 0
    brier: float = 0.0
    nll: float = 0.0
    per_class_f1: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class BenchmarkResult:
    cards: list[MetricsCard]
    best_backend: str
    comparison: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cards"] = [c.to_dict() for c in self.cards]
        return d

class BenchmarkRunner:
    def evaluate_backend(
        self,
        backend: str,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        logits: np.ndarray | None = None,
    ) -> MetricsCard:
        if X is None or y is None or logits is None:
            return MetricsCard(backend=backend, n_eval=0)

        # Compute probabilities via softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        preds = np.argmax(probs, axis=1)
        accuracy = float(np.mean(preds == y))

        # F1 Macro & per-class F1
        num_classes = probs.shape[1]
        per_class_f1 = {}
        f1_list = []
        for c in range(num_classes):
            tp = np.sum((preds == c) & (y == c))
            fp = np.sum((preds == c) & (y != c))
            fn = np.sum((preds != c) & (y == c))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            per_class_f1[c] = float(f1)
            f1_list.append(f1)
        f1_macro = float(np.mean(f1_list))

        # ECE
        ece = self._ece(probs, y)

        # Brier score
        C = probs.shape[1]
        y_one_hot = np.eye(C)[y]
        brier = float(np.mean(np.sum((probs - y_one_hot) ** 2, axis=1)))

        # NLL
        nll = float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-15, 1.0))))

        return MetricsCard(
            backend=backend,
            accuracy=accuracy,
            f1_macro=f1_macro,
            ece=ece,
            n_eval=len(y),
            brier=brier,
            nll=nll,
            per_class_f1=per_class_f1,
        )

    def _ece(self, probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
        preds = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        accuracies = (preds == y)
        
        ece = 0.0
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            if i == n_bins - 1:
                in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
            else:
                in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
                
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
                
        return float(ece)

    def compare(self, cards: list[MetricsCard]) -> BenchmarkResult:
        if not cards:
            return BenchmarkResult(cards=[], best_backend="")

        best_card = max(cards, key=lambda c: (c.accuracy, c.f1_macro))

        comparison = {}
        for i in range(len(cards)):
            for j in range(i + 1, len(cards)):
                c1 = cards[i]
                c2 = cards[j]
                key = f"{c1.backend}_vs_{c2.backend}"
                comparison[key] = {
                    "accuracy_diff": float(c1.accuracy - c2.accuracy),
                    "f1_macro_diff": float(c1.f1_macro - c2.f1_macro),
                    "ece_diff": float(c1.ece - c2.ece),
                    "nll_diff": float(c1.nll - c2.nll),
                    "brier_diff": float(c1.brier - c2.brier),
                }

        return BenchmarkResult(cards=cards, best_backend=best_card.backend, comparison=comparison)
