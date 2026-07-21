import torch
from pathlib import Path

def save_model_checkpoint(config, model, optimizer, scheduler, scaler, epoch, best_metric):
    """Saves model checkpoints, optimizer state, scheduler state, and history elements."""
    checkpoint_state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "arch": "densenet121",
    }
    
    # Save last checkpoint
    torch.save(checkpoint_state, config.last_checkpoint_path)
    
    # Save optimizer and scheduler separately as requested by the user
    torch.save(optimizer.state_dict(), config.out_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), config.out_dir / "scheduler.pt")
    
def save_best_model(config, model, epoch, best_metric):
    """Saves the best model weights."""
    checkpoint_state = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "arch": "densenet121",
    }
    torch.save(checkpoint_state, config.best_model_path)
    print(f"[Checkpoint] Saved new best model to {config.best_model_path.name} (Macro-AUROC: {best_metric:.4f})")

def load_model_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None, device="cpu"):
    """Loads a model checkpoint and resumes optimizer, scheduler, and scaler states."""
    # weights_only=False is required to restore optimizer/scheduler/scaler state
    # (not plain tensors). This is a training-only resume path that loads a
    # checkpoint this same process wrote — not an untrusted artifact — so the safe
    # unpickler is not applicable here (cf. the serving load in inference.py which
    # uses weights_only=True). See audit §11.5.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
    if scaler and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
    epoch = checkpoint.get("epoch", -1)
    best_metric = checkpoint.get("best_metric", 0.0)
    print(f"[Checkpoint] Resumed from epoch {epoch} with best Macro-AUROC {best_metric:.4f}")
    return epoch + 1, best_metric
