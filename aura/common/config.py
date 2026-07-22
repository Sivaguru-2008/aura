"""Central configuration. Reads [tool.aura] from pyproject.toml, overridable by env.

Single knob surface so behaviour (e.g. fusion backend) is swappable without code
changes — the modularity requirement made operational.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the aura/ directory
ARTIFACTS = ROOT / "artifacts"                          # trained params, calibration
DATA = ROOT / "data"
DB_PATH = ARTIFACTS / "aura.db"

_PRESENT_THR: dict | None = None


def finding_present_threshold(finding_value: str, default: float = 0.5) -> float:
    """Per-finding operating point for the 'present' decision.

    Once vision probabilities are Platt-calibrated (honest, low — matching the
    2–18% true prevalence), a fixed 0.5 cutoff under-calls every finding. The
    F1-optimal per-finding thresholds fitted on held-out patients live in
    ``vision_serving_calibration.json``; fall back to ``default`` when absent so
    the uncalibrated path is unchanged.
    """
    global _PRESENT_THR
    if _PRESENT_THR is None:
        import json
        p = ARTIFACTS / "vision_serving_calibration.json"
        try:
            _PRESENT_THR = json.loads(p.read_text()).get("per_finding_threshold", {}) if p.exists() else {}
        except Exception:
            _PRESENT_THR = {}
    return float(_PRESENT_THR.get(finding_value, default))


@dataclass(frozen=True)
class Settings:
    fusion_backend: str = "quantum"          # "quantum" | "classical"
    # Vision backbone: "features" (numpy fallback), "densenet_mimic" (real
    # MIMIC-CXR DenseNet-121), or "timm" (fine-tuned EfficientNetV2/ConvNeXt/Swin).
    vision_backend: str = "features"
    vision_arch: str = "densenet121"         # timm arch key when vision_backend=timm
    vision_weights: str = "densenet121-res224-mimic_ch"   # torchxrayvision weight tag
    n_qubits: int = 8                        # evidence-vector width
    n_layers: int = 3                        # VQC depth
    n_shots: int = 512                       # finite-shot readout (uncertainty)
    ensemble_size: int = 5                   # deep-ensemble members
    conformal_coverage: float = 0.90
    # Abstention operating point. Recalibrated to the real MIMIC evidence
    # distribution (audit F1/F8): the previous values (0.45, 3) were tuned for the
    # overconfident *synthetic* fusion model (temperature 0.77, sharpening) and
    # abstained on ~91% of real films once fusion was honestly calibrated
    # (temperature 0.94). At 90% coverage a genuinely-uncertain 6-class model needs
    # sets of 4–6; we abstain only when it cannot narrow below 5 (set > 4) or the
    # top calibrated probability is below 0.30 (a weak call for 6 classes). This
    # commits on the clearer ~1/3 and defers the ambiguous majority to a human —
    # the intended "calibrated doubt" behaviour, now on real data.
    abstention_conformal_size: int = 4
    ood_energy_threshold: float = 1.5
    low_confidence_threshold: float = 0.30   # abstain if top calibrated prob below
    seed: int = 7
    # --- Module 5: quantum-classical conflict guard (Wasserstein tie-breaker) --- #
    fusion_conflict_guard: bool = True       # defer to PoE when VQC diverges
    fusion_conflict_tau: float = 0.12        # static floor for the dynamic threshold τ
    # --- Module 8: adaptive conformal inference (ACI) --------------------------- #
    aci_enabled: bool = True                 # update q̂ online from confirmed outcomes
    aci_gamma: float = 0.02                  # ACI step size γ (learning rate on q̂)
    aci_window: int = 200                    # rolling window for localized coverage
    # --- Fusion training data source (audit F1) --------------------------------- #
    # "mimic" trains fusion on real DenseNet evidence over real films (matches
    # serving); falls back to "synthetic" automatically when the corpus is absent.
    fusion_train_source: str = "mimic"       # "mimic" | "synthetic"
    fusion_train_n: int = 900                # target #real studies for the fusion set
    # --- Module 2: vision conditioning ----------------------------------------- #
    vision_tv_weight: float = 1e-4           # total-variation reg weight on latent maps
    # --- Security (all opt-in; defaults preserve the offline P0 demo) ----------- #
    max_upload_mb: float = 25.0              # reject uploads larger than this (DoS guard)
    auth_token: str = ""                     # when set, mutating endpoints require it
    auth_header: str = "x-aura-token"        # header carrying the shared token
    rate_limit_rpm: int = 0                  # >0 enables a per-client request/minute cap


def _as_bool(v) -> bool:
    """Cast pyproject/env values to bool. Env vars arrive as strings."""
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _from_pyproject() -> dict:
    pp = ROOT / "pyproject.toml"
    if not pp.exists():
        return {}
    with pp.open("rb") as f:
        data = tomllib.load(f)
    return data.get("tool", {}).get("aura", {})


@lru_cache
def get_settings() -> Settings:
    base = Settings()
    over = _from_pyproject()
    # env vars win last (AURA_FUSION_BACKEND, AURA_N_SHOTS, ...)
    def pick(name: str, cast, default):
        env = os.environ.get(f"AURA_{name.upper()}")
        if env is not None:
            return cast(env)
        if name in over:
            return cast(over[name])
        return default

    return Settings(
        fusion_backend=pick("fusion_backend", str, base.fusion_backend),
        vision_backend=pick("vision_backend", str, base.vision_backend),
        vision_arch=pick("vision_arch", str, base.vision_arch),
        vision_weights=pick("vision_weights", str, base.vision_weights),
        n_qubits=pick("n_qubits", int, base.n_qubits),
        n_layers=pick("n_layers", int, base.n_layers),
        n_shots=pick("n_shots", int, base.n_shots),
        ensemble_size=pick("ensemble_size", int, base.ensemble_size),
        conformal_coverage=pick("conformal_coverage", float, base.conformal_coverage),
        abstention_conformal_size=pick(
            "abstention_conformal_size", int, base.abstention_conformal_size
        ),
        ood_energy_threshold=pick("ood_energy_threshold", float, base.ood_energy_threshold),
        low_confidence_threshold=pick(
            "low_confidence_threshold", float, base.low_confidence_threshold
        ),
        seed=pick("seed", int, base.seed),
        fusion_conflict_guard=pick("fusion_conflict_guard", _as_bool, base.fusion_conflict_guard),
        fusion_conflict_tau=pick("fusion_conflict_tau", float, base.fusion_conflict_tau),
        aci_enabled=pick("aci_enabled", _as_bool, base.aci_enabled),
        aci_gamma=pick("aci_gamma", float, base.aci_gamma),
        aci_window=pick("aci_window", int, base.aci_window),
        vision_tv_weight=pick("vision_tv_weight", float, base.vision_tv_weight),
        max_upload_mb=pick("max_upload_mb", float, base.max_upload_mb),
        auth_token=pick("auth_token", str, base.auth_token),
        auth_header=pick("auth_header", str, base.auth_header),
        rate_limit_rpm=pick("rate_limit_rpm", int, base.rate_limit_rpm),
        fusion_train_source=pick("fusion_train_source", str, base.fusion_train_source),
        fusion_train_n=pick("fusion_train_n", int, base.fusion_train_n),
    )


def ensure_dirs() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
