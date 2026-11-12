"""Local cohort validation: how the deployed pipeline did on this site's verified outcomes.

The input is the only ground truth the product holds: cases for which a clinician
recorded a verified outcome (PCR, biopsy, pathology, expert consensus, clinical course;
see ``POST /v1/cases/{id}/outcome``), each paired with the posterior AURA stored when it
decided the case. Nothing is simulated and nothing is defaulted. Below
``MIN_VERIFIED_OUTCOMES`` the function refuses rather than report a number that means
nothing: a validation report with invented figures is worse than no report.
"""
from __future__ import annotations

import math
from typing import Iterable

#: Fewer verified outcomes than this and every metric's interval is too wide to act on.
MIN_VERIFIED_OUTCOMES = 20


class InsufficientOutcomes(ValueError):
    """Too few verified outcomes for any metric to mean something."""


def _auroc(pos: list[float], neg: list[float]) -> float:
    """Mann-Whitney AUROC with average ranks for ties."""
    scored = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg], key=lambda t: t[0])
    ranks = [0.0] * len(scored)
    i = 0
    while i < len(scored):
        j = i
        while j + 1 < len(scored) and scored[j + 1][0] == scored[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    rank_sum = sum(r for r, (_, label) in zip(ranks, scored) if label == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def compute(pairs: Iterable[dict], *, min_n: int = MIN_VERIFIED_OUTCOMES,
            n_bins: int = 10) -> dict:
    """Metrics over ``{"posterior": {dx: p}, "truth": dx, "covered": bool, "abstained": bool}``.

    Returns ``{"metrics": {...}, "failure_analysis": {...}}``. Raises
    :class:`InsufficientOutcomes` when fewer than ``min_n`` records carry both a stored
    posterior and a verified diagnosis.
    """
    usable = [p for p in pairs if p.get("posterior") and p.get("truth")]
    n = len(usable)
    if n < min_n:
        raise InsufficientOutcomes(
            f"{n} case(s) have both a verified outcome and a stored posterior; at least "
            f"{min_n} are needed before a validation run means anything. Record verified "
            f"outcomes with POST /v1/cases/{{case_id}}/outcome.")

    labels = sorted({dx for p in usable for dx in p["posterior"]} | {p["truth"] for p in usable})
    top = [max(p["posterior"], key=p["posterior"].get) for p in usable]
    correct = [t == p["truth"] for t, p in zip(top, usable)]
    conf = [float(p["posterior"][t]) for t, p in zip(top, usable)]
    k = sum(correct)

    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, c in enumerate(conf) if lo < c <= hi or (b == 0 and c <= 0.0)]
        if idx:
            acc_b = sum(correct[i] for i in idx) / len(idx)
            conf_b = sum(conf[i] for i in idx) / len(idx)
            ece += len(idx) / n * abs(acc_b - conf_b)

    brier = sum(
        sum((p["posterior"].get(dx, 0.0) - (1.0 if dx == p["truth"] else 0.0)) ** 2
            for dx in labels)
        for p in usable) / n

    per_class: dict[str, float] = {}
    for dx in labels:
        pos = [p["posterior"].get(dx, 0.0) for p in usable if p["truth"] == dx]
        neg = [p["posterior"].get(dx, 0.0) for p in usable if p["truth"] != dx]
        if pos and neg:
            per_class[dx] = round(_auroc(pos, neg), 4)

    covered = sum(bool(p.get("covered")) for p in usable)
    abstained = sum(bool(p.get("abstained")) for p in usable)
    metrics = {
        "n": n,
        "accuracy": round(k / n, 4),
        "accuracy_ci95": list(_wilson(k, n)),
        "macro_auroc": (round(sum(per_class.values()) / len(per_class), 4)
                        if per_class else None),
        "per_class_auroc": per_class,
        "classes_scored": len(per_class),
        "ece": round(ece, 4),
        "brier_score": round(brier, 4),
        "conformal_coverage": round(covered / n, 4),
        "abstention_rate": round(abstained / n, 4),
    }
    failure = {
        "incorrect_top1": n - k,
        "abstained": abstained,
        "truth_outside_conformal_set": n - covered,
    }
    return {"metrics": metrics, "failure_analysis": failure}


def render_report(run: dict, tenant_id: str, metrics: dict, basis: str) -> str:
    """Markdown report for a completed run. Prints only what is in ``metrics``."""
    lines = [
        f"# Local Cohort Validation Report: {run['model_id']}",
        "",
        f"- Dataset: {run['dataset_name']}",
        f"- Organization: {tenant_id}",
        f"- Basis: {basis}",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for key, value in metrics.items():
        if key in ("per_class_auroc",):
            continue
        lines.append(f"| {key} | {value} |")
    if metrics.get("per_class_auroc"):
        lines += ["", "Per-class AUROC (one-vs-rest; classes with both outcomes present):", ""]
        lines += [f"- {dx}: {v}" for dx, v in metrics["per_class_auroc"].items()]
    return "\n".join(lines) + "\n"
