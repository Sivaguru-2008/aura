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


@dataclass(frozen=True)
class Settings:
    fusion_backend: str = "quantum"          # "quantum" | "classical"
    n_qubits: int = 8                        # evidence-vector width
    n_layers: int = 3                        # VQC depth
    n_shots: int = 512                       # finite-shot readout (uncertainty)
    ensemble_size: int = 5                   # deep-ensemble members
    conformal_coverage: float = 0.90
    abstention_conformal_size: int = 3
    ood_energy_threshold: float = 1.5
    low_confidence_threshold: float = 0.45   # abstain if top calibrated prob below
    seed: int = 7


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
    )


def ensure_dirs() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
