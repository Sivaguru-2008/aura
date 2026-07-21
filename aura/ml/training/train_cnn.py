"""Production training harness for the CNN vision backbone (GPU).

Implements the full training stack the project lacked:

  * GPU DataLoader + Albumentations aug + class-balanced weighted sampling + decode cache
  * Mixed precision (torch.amp) with GradScaler
  * Gradient accumulation (large effective batch on an 8 GB card)
  * Cosine LR schedule with warmup-free restart-friendly resume
  * Checkpointing + resume (model / optimizer / scheduler / scaler / epoch / best)
  * Early stopping on validation macro-AUROC
  * TensorBoard logging (loss, LR, per-epoch metrics)

Runs on the synthetic world out of the box (``--synthetic``); point ``manifest``
at a real MIMIC-CXR/CheXpert label CSV to fine-tune ``timm`` backbones
(DenseNet-121 / EfficientNetV2 / ConvNeXt / Swin) on real radiographs. The best
checkpoint is written to ``artifacts/vision_cnn.pt`` — exactly what
``CXRBackbone(kind="timm")`` loads at serving time.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from common.config import ARTIFACTS, ensure_dirs
from schemas.clinical import FINDINGS
from services.vision.cnn import TIMM_ARCHES, select_device
from ml.training.cxr_dataset import (
    CXRManifestDataset,
    SyntheticCXRDataset,
    make_loader,
)


@dataclass
class TrainConfig:
    arch: str = "densenet121"          # key in services.vision.cnn.TIMM_ARCHES
    epochs: int = 15
    batch: int = 16
    accum_steps: int = 2               # effective batch = batch * accum_steps
    lr: float = 3e-4
    weight_decay: float = 1e-4
    amp: bool = True
    patience: int = 4                  # early-stopping patience (epochs)
    balanced: bool = True
    out_dir: str = str(ARTIFACTS)
    ckpt_name: str = "vision_cnn.pt"
    resume: bool = True
    seed: int = 7


class Trainer:
    def __init__(self, cfg: TrainConfig):
        import torch

        self.cfg = cfg
        self.device = select_device()
        self.device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        torch.manual_seed(cfg.seed)
        self.best = -1.0
        self.start_epoch = 0

        import timm
        self.model = timm.create_model(
            TIMM_ARCHES.get(cfg.arch, cfg.arch),
            pretrained=True, num_classes=len(FINDINGS), in_chans=1,
        ).to(self.device)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=cfg.epochs)
        self.scaler = torch.amp.GradScaler(enabled=(cfg.amp and self.device_type == "cuda"))
        self.writer = self._make_writer()
        self.ckpt_path = Path(cfg.out_dir) / cfg.ckpt_name
        self.last_path = Path(cfg.out_dir) / (Path(cfg.ckpt_name).stem + "_last.pt")

    def _make_writer(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
            return SummaryWriter(log_dir=str(Path(self.cfg.out_dir) / "tb" / "vision_cnn"))
        except Exception as e:
            print(f"[train_cnn] TensorBoard unavailable ({e}); logging to stdout only.")
            return None

    # ---- checkpointing ---------------------------------------------------- #
    def _pos_weight(self, labels: np.ndarray):
        import torch
        pos = labels.sum(axis=0)
        neg = len(labels) - pos
        pw = np.clip(neg / np.clip(pos, 1.0, None), 0.1, 20.0)
        return torch.tensor(pw, dtype=torch.float32, device=self.device)

    def save(self, epoch: int, metric: float, best: bool):
        import torch
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scheduler": self.sched.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "best": self.best,
            "arch": self.cfg.arch,
            "findings": [f.value for f in FINDINGS],
            "version": f"vision-timm-{self.cfg.arch}-e{epoch}",
        }
        torch.save(state, self.last_path)
        if best:
            torch.save(state, self.ckpt_path)

    def maybe_resume(self):
        import torch
        if not (self.cfg.resume and self.last_path.exists()):
            return
        # weights_only=False: restores optimizer/scheduler state from a checkpoint
        # this process wrote (trusted, training-only resume — audit §11.5).
        s = torch.load(self.last_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(s["model"])
        self.opt.load_state_dict(s["optimizer"])
        self.sched.load_state_dict(s["scheduler"])
        self.scaler.load_state_dict(s["scaler"])
        self.start_epoch = s["epoch"] + 1
        self.best = s.get("best", -1.0)
        print(f"[train_cnn] resumed from epoch {s['epoch']} (best AUROC {self.best:.4f})")

    # ---- loops ------------------------------------------------------------ #
    def _run_epoch(self, loader, loss_fn, train: bool):
        import torch
        self.model.train(train)
        total, n = 0.0, 0
        all_logits, all_y = [], []
        self.opt.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            with torch.amp.autocast(device_type=self.device_type,
                                    enabled=(self.cfg.amp and self.device_type == "cuda")):
                logits = self.model(x)
                loss = loss_fn(logits, y) / self.cfg.accum_steps
            if train:
                self.scaler.scale(loss).backward()
                if (step + 1) % self.cfg.accum_steps == 0:
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad(set_to_none=True)
            total += float(loss.detach()) * self.cfg.accum_steps * len(x)
            n += len(x)
            all_logits.append(logits.detach().float().cpu().numpy())
            all_y.append(y.detach().cpu().numpy())
        return total / n, np.concatenate(all_logits), np.concatenate(all_y)

    def _macro_auroc(self, logits, y) -> float:
        from sklearn.metrics import roc_auc_score
        probs = 1.0 / (1.0 + np.exp(-logits))
        aucs = []
        for c in range(y.shape[1]):
            if 0 < y[:, c].sum() < len(y):
                aucs.append(roc_auc_score(y[:, c], probs[:, c]))
        return float(np.mean(aucs)) if aucs else float("nan")

    def fit(self, train_ds, val_ds):
        import torch
        ensure_dirs()
        self.maybe_resume()
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self._pos_weight(train_ds.labels))
        tl = make_loader(train_ds, self.cfg.batch, True, self.device, self.cfg.balanced)
        vl = make_loader(val_ds, self.cfg.batch, False, self.device)

        bad = 0
        for epoch in range(self.start_epoch, self.cfg.epochs):
            t0 = time.time()
            tr_loss, _, _ = self._run_epoch(tl, loss_fn, train=True)
            with torch.no_grad():
                va_loss, vlog, vy = self._run_epoch(vl, loss_fn, train=False)
            self.sched.step()
            auroc = self._macro_auroc(vlog, vy)
            lr = self.opt.param_groups[0]["lr"]
            print(f"[train_cnn] epoch {epoch:2d} | train {tr_loss:.4f} | val {va_loss:.4f} "
                  f"| val_macroAUROC {auroc:.4f} | lr {lr:.2e} | {time.time()-t0:.1f}s")
            if self.writer:
                self.writer.add_scalar("loss/train", tr_loss, epoch)
                self.writer.add_scalar("loss/val", va_loss, epoch)
                self.writer.add_scalar("metric/val_macro_auroc", auroc, epoch)
                self.writer.add_scalar("lr", lr, epoch)

            is_best = auroc > self.best
            if is_best:
                self.best, bad = auroc, 0
            else:
                bad += 1
            self.save(epoch, auroc, is_best)
            if bad >= self.cfg.patience:
                print(f"[train_cnn] early stopping at epoch {epoch} "
                      f"(no improvement for {self.cfg.patience} epochs)")
                break

        if self.writer:
            self.writer.close()
        return {"best_macro_auroc": round(self.best, 4), "checkpoint": str(self.ckpt_path)}


def run(manifest: str | None = None, synthetic: bool = True, n: int = 600,
        cfg: TrainConfig | None = None) -> dict:
    cfg = cfg or TrainConfig()
    if manifest:
        rows = _read_manifest(manifest)
        k = int(0.85 * len(rows))
        train_ds = CXRManifestDataset(rows[:k], train=True)
        val_ds = CXRManifestDataset(rows[k:], train=False)
    else:
        train_ds = SyntheticCXRDataset(int(n * 0.85), seed=cfg.seed, train=True)
        val_ds = SyntheticCXRDataset(int(n * 0.15), seed=cfg.seed + 1, train=False)
    print(f"[train_cnn] arch={cfg.arch} device={select_device()} "
          f"train={len(train_ds)} val={len(val_ds)} amp={cfg.amp}")
    res = Trainer(cfg).fit(train_ds, val_ds)
    (Path(cfg.out_dir) / "vision_cnn_train.json").write_text(
        json.dumps({**res, "config": asdict(cfg)}, indent=2))
    print(f"[train_cnn] done — best val macro-AUROC {res['best_macro_auroc']}")
    return res


def _read_manifest(path: str) -> list[dict]:
    """CSV/JSONL -> [{path, labels:[0/1]*len(FINDINGS)}]. CSV columns: path,<finding cols>."""
    p = Path(path)
    rows: list[dict] = []
    if p.suffix.lower() == ".jsonl":
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    import csv
    with p.open() as f:
        for r in csv.DictReader(f):
            labels = [float(r.get(fd.value, 0) or 0) >= 0.5 and 1.0 or 0.0 for fd in FINDINGS]
            rows.append({"path": r["path"], "labels": [float(v) for v in labels]})
    return rows


if __name__ == "__main__":
    run()
