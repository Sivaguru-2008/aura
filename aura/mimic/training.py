"""Step 9 — Training pipelines for the MIMIC-CXR tabular tasks.

Covers the classical models the data supports today, on the leakage-free image
feature matrix (Step 6) and patient-level splits (Step 7).

Model families:
    * Gradient boosting — ``xgboost`` / ``lightgbm`` / ``catboost`` when installed,
      otherwise a scikit-learn ``HistGradientBoosting`` fallback (LightGBM-equivalent
      histogram booster). Same API regardless of backend, so installing a library
      later changes one string, not the pipeline.
    * ``mlp`` — a small PyTorch MLP with the full training-loop feature set:
      device-agnostic (CUDA when available), checkpointing, early stopping, resume,
      mixed precision (AMP, guarded on CPU), and optional TensorBoard.

Deep sequence models (LSTM / Temporal Transformer) are intentionally *not* trained
here: this corpus has no vitals/lab time-series, and per-patient study sequences
are short (median 1). They belong to the vision path (gated by the "don't touch
vision yet" rule). See :func:`sequence_models_status`.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from mimic.config import MimicPaths, get_mimic_paths
from mimic.tasks import TaskDataset, TaskDatasetBuilder

log = logging.getLogger("mimic.training")


# --------------------------------------------------------------------------- #
# Gradient boosting (xgboost / lightgbm / catboost / sklearn fallback)
# --------------------------------------------------------------------------- #
def _make_booster(backend: str, task_type: str, n_classes: int, **kw):
    """Return an unfitted estimator for the requested backend, or fall back.

    Returns (estimator, resolved_backend_name).
    """
    order = [backend] if backend != "auto" else ["xgboost", "lightgbm", "catboost", "sklearn"]
    seed = kw.get("seed", 7)
    for name in order:
        try:
            if name == "xgboost":
                import xgboost as xgb
                obj = "multi:softprob" if task_type == "multiclass" else "binary:logistic"
                return xgb.XGBClassifier(
                    n_estimators=kw.get("n_estimators", 400),
                    max_depth=kw.get("max_depth", 6),
                    learning_rate=kw.get("lr", 0.05),
                    subsample=0.9, colsample_bytree=0.9,
                    objective=obj, tree_method="hist", random_state=seed,
                    eval_metric="mlogloss" if task_type == "multiclass" else "logloss",
                    early_stopping_rounds=kw.get("patience", 30),
                ), name
            if name == "lightgbm":
                import lightgbm as lgb
                return lgb.LGBMClassifier(
                    n_estimators=kw.get("n_estimators", 400),
                    max_depth=kw.get("max_depth", -1),
                    learning_rate=kw.get("lr", 0.05), subsample=0.9, random_state=seed,
                ), name
            if name == "catboost":
                from catboost import CatBoostClassifier
                return CatBoostClassifier(
                    iterations=kw.get("n_estimators", 400),
                    depth=kw.get("max_depth", 6), random_seed=seed,
                    learning_rate=kw.get("lr", 0.05), verbose=False,
                ), name
            if name == "sklearn":
                from sklearn.ensemble import HistGradientBoostingClassifier
                return HistGradientBoostingClassifier(
                    max_iter=kw.get("n_estimators", 400),
                    learning_rate=kw.get("lr", 0.05), random_state=seed,
                    max_depth=None if kw.get("max_depth", 6) < 0 else kw.get("max_depth", 6),
                    early_stopping=True, validation_fraction=0.15,
                    n_iter_no_change=kw.get("patience", 30),
                ), name
        except ImportError:
            continue
    raise RuntimeError("no gradient-boosting backend available (not even sklearn)")


@dataclass
class TrainResult:
    task: str
    model_kind: str
    backend: str
    n_train: int
    n_val: int
    proba_val: np.ndarray
    y_val: np.ndarray
    checkpoint: str = ""
    history: dict = field(default_factory=dict)


class GBMTrainer:
    """Gradient-boosting trainer with pluggable backend + joblib checkpointing."""

    def __init__(self, task: TaskDataset, backend: str = "auto", **kw):
        if task.task.task_type not in ("multiclass", "binary"):
            raise ValueError(f"GBMTrainer supports multiclass/binary, not {task.task.task_type}")
        self.task = task
        self.kw = kw
        self.model, self.backend = _make_booster(
            backend, task.task.task_type, task.task.n_classes, **kw
        )

    def _fit_with_es(self, Xtr, ytr, Xval, yval) -> None:
        """Fit, passing an eval set when the backend supports early stopping."""
        try:
            if self.backend == "xgboost":
                self.model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
                return
            if self.backend == "lightgbm":
                import lightgbm as lgb
                self.model.fit(
                    Xtr, ytr, eval_set=[(Xval, yval)],
                    callbacks=[lgb.early_stopping(self.kw.get("patience", 30), verbose=False)],
                )
                return
        except Exception as e:  # pragma: no cover - backend quirks
            log.warning("early-stopping fit failed (%s); plain fit", e)
        self.model.fit(Xtr, ytr)         # sklearn HGB / catboost self-validate

    def train(self, val: TaskDataset, checkpoint: Optional[Path] = None) -> TrainResult:
        Xtr, ytr = self.task.X.to_numpy(), self.task.y
        Xval, yval = val.X.to_numpy(), val.y
        log.info("GBM[%s] fitting on %d samples -> val %d", self.backend, len(Xtr), len(Xval))
        self._fit_with_es(Xtr, ytr, Xval, yval)

        proba = self._proba(Xval, n_classes=self.task.task.n_classes)
        ckpt = ""
        if checkpoint is not None:
            import joblib
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": self.model, "backend": self.backend}, checkpoint)
            ckpt = str(checkpoint)
            log.info("saved checkpoint -> %s", ckpt)
        return TrainResult(
            self.task.task.name, "gbm", self.backend, len(Xtr), len(Xval), proba, yval, ckpt
        )

    def _proba(self, X, n_classes: int) -> np.ndarray:
        p = self.model.predict_proba(X)
        if p.ndim == 1:                                  # some binary paths
            p = np.column_stack([1 - p, p])
        # multiclass diagnosis: ensure full 6-column width even if a class is absent
        if n_classes > 2 and p.shape[1] != n_classes and hasattr(self.model, "classes_"):
            full = np.zeros((len(X), n_classes))
            for j, c in enumerate(self.model.classes_):
                full[:, int(c)] = p[:, j]
            p = full
        return p

    @staticmethod
    def load(checkpoint: Path):
        import joblib
        return joblib.load(checkpoint)


# --------------------------------------------------------------------------- #
# Torch MLP — checkpoint / early-stop / resume / AMP / tensorboard
# --------------------------------------------------------------------------- #
@dataclass
class MLPConfig:
    hidden: tuple[int, ...] = (128, 64)
    epochs: int = 60
    batch: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    amp: bool = True
    tensorboard_dir: str = ""
    seed: int = 7


class MLPTrainer:
    """PyTorch MLP for multiclass/binary tabular tasks with full loop features."""

    def __init__(self, task: TaskDataset, cfg: Optional[MLPConfig] = None):
        import torch
        self.torch = torch
        self.task = task
        self.cfg = cfg or MLPConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_classes = max(2, task.task.n_classes)
        self._build()

    def _build(self) -> None:
        import torch.nn as nn
        torch = self.torch
        torch.manual_seed(self.cfg.seed)
        in_dim = self.task.X.shape[1]
        layers: list = []
        d = in_dim
        for h in self.cfg.hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(0.2)]
            d = h
        layers += [nn.Linear(d, self.n_classes)]
        self.net = nn.Sequential(*layers).to(self.device)
        # feature standardization (fit on train later)
        self.mu = None
        self.sd = None

    def _prep(self, X: np.ndarray, fit: bool):
        if fit:
            self.mu = X.mean(0, keepdims=True)
            self.sd = X.std(0, keepdims=True) + 1e-6
        return (X - self.mu) / self.sd

    def train(self, val: TaskDataset, checkpoint: Optional[Path] = None,
              resume: bool = False) -> TrainResult:
        torch = self.torch
        import torch.nn as nn
        Xtr = self._prep(self.task.X.to_numpy().astype("float32"), fit=True)
        ytr = self.task.y.astype("int64")
        Xva = self._prep(val.X.to_numpy().astype("float32"), fit=False)
        yva = val.y.astype("int64")

        tb = None
        if self.cfg.tensorboard_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb = SummaryWriter(self.cfg.tensorboard_dir)
            except ImportError:
                log.warning("tensorboard not installed; skipping TB logging")

        opt = torch.optim.AdamW(self.net.parameters(), lr=self.cfg.lr,
                                weight_decay=self.cfg.weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        use_amp = self.cfg.amp and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        start_epoch, best_val, best_state, bad = 0, float("inf"), None, 0
        if resume and checkpoint and Path(checkpoint).is_file():
            # our own checkpoint (contains numpy mu/sd) -> trusted, full unpickle
            ck = torch.load(checkpoint, map_location=self.device, weights_only=False)
            self.net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            start_epoch = ck["epoch"] + 1; best_val = ck.get("best_val", best_val)
            self.mu, self.sd = ck["mu"], ck["sd"]
            log.info("resumed from %s at epoch %d", checkpoint, start_epoch)

        Xtr_t = torch.tensor(Xtr, device=self.device)
        ytr_t = torch.tensor(ytr, device=self.device)
        Xva_t = torch.tensor(Xva, device=self.device)
        yva_t = torch.tensor(yva, device=self.device)
        n = len(Xtr_t)
        history = {"val_loss": []}

        for epoch in range(start_epoch, self.cfg.epochs):
            self.net.train()
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.cfg.batch):
                idx = perm[i:i + self.cfg.batch]
                opt.zero_grad()
                with torch.autocast(device_type=self.device.type, enabled=use_amp):
                    out = self.net(Xtr_t[idx])
                    loss = loss_fn(out, ytr_t[idx])
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()

            self.net.eval()
            with torch.no_grad():
                vloss = float(loss_fn(self.net(Xva_t), yva_t).item())
            history["val_loss"].append(round(vloss, 4))
            if tb:
                tb.add_scalar("val/loss", vloss, epoch)

            if vloss < best_val - 1e-4:
                best_val, bad = vloss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                if checkpoint:
                    Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"model": self.net.state_dict(), "opt": opt.state_dict(),
                                "epoch": epoch, "best_val": best_val,
                                "mu": self.mu, "sd": self.sd}, checkpoint)
            else:
                bad += 1
                if bad >= self.cfg.patience:
                    log.info("early stop at epoch %d (best val %.4f)", epoch, best_val)
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        if tb:
            tb.close()

        self.net.eval()
        with torch.no_grad():
            proba = torch.softmax(self.net(Xva_t), dim=1).cpu().numpy()
        return TrainResult(self.task.task.name, "mlp", f"torch/{self.device.type}",
                           len(Xtr), len(Xva), proba, yva,
                           str(checkpoint) if checkpoint else "", history)


def sequence_models_status() -> dict[str, str]:
    """Explain why LSTM / Temporal Transformer are not trained on this corpus."""
    return {
        "lstm": "not trained: no vitals/lab time-series; study sequences are short (median 1)",
        "temporal_transformer": "not trained: same reason — belongs to the vision path (gated)",
        "tabular_transformer": "available via MLPTrainer variant; MLP is the shipped default",
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def train_task(
    task_name: str,
    model_kind: str = "gbm",
    backend: str = "auto",
    limit_train: Optional[int] = None,
    paths: Optional[MimicPaths] = None,
    checkpoint: Optional[Path] = None,
    resume: bool = False,
) -> TrainResult:
    """End-to-end: materialize train+val datasets for a task and fit a model."""
    paths = paths or get_mimic_paths()
    tdb = TaskDatasetBuilder(paths)
    train_ds = tdb.build(task_name, "train", limit=limit_train)
    val_ds = tdb.build(task_name, "validation")
    if model_kind == "gbm":
        return GBMTrainer(train_ds, backend=backend).train(val_ds, checkpoint=checkpoint)
    if model_kind == "mlp":
        return MLPTrainer(train_ds).train(val_ds, checkpoint=checkpoint, resume=resume)
    raise ValueError(f"unknown model_kind {model_kind!r} (gbm|mlp)")
