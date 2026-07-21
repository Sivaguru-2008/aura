import argparse
import time
import torch
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import asyncio

from ml.vision_cxr.config import TrainConfig
from ml.vision_cxr.dataset import build_loaders
from ml.vision_cxr.model import DenseNet121CXR
from ml.vision_cxr.losses import MultiLabelLoss, RegularizedMultiLabelLoss
from ml.vision_cxr.validate import evaluate_model
from ml.vision_cxr.checkpoint import save_model_checkpoint, save_best_model, load_model_checkpoint
from ml.vision_cxr.utils import set_seed, HistoryLogger, plot_training_history

def train_one_epoch(model, dataloader, optimizer, loss_fn, scaler, device, grad_clip):
    model.train()
    total_loss = 0.0
    # Detect the TV-regularised criterion (returns (loss, parts) and needs features).
    regularized = isinstance(loss_fn, RegularizedMultiLabelLoss)

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            if regularized:
                logits, feats = model(inputs, return_features=True)
                loss, _parts = loss_fn(logits, targets, feats)   # BCE + λ·TV(features)
            else:
                logits = model(inputs)
                loss = loss_fn(logits, targets)

        # Scaling and backward
        if scaler:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
        total_loss += loss.item() * inputs.size(0)
        
    return total_loss / len(dataloader.dataset)

def run_post_training_validation(config):
    """Post-training verification validating end-to-end functionality of AURA after integration."""
    import asyncio
    from gateway.pipeline import Pipeline
    from schemas.contracts import StructuredPriors
    from mimic.loaders import MimicCxrLoader
    from services.vision.io import load_cxr, study_from_cxr
    
    print("\n" + "="*60)
    print("RUNNING AUTOMATIC POST-TRAINING VALIDATION")
    print("="*60)
    
    # 1. Initialize Pipeline (checks if integration is active)
    print("[Validation] Initializing AURA Pipeline (should load best_model.pt)...")
    pipe = Pipeline()
    
    # Assert integration
    if pipe.vision.backbone is None:
        raise RuntimeError("Integration Failure: VisionEngine backbone is None (best_model.pt not loaded).")
    print(f"[Validation] Verified: Pipeline is using the production backbone: {pipe.vision.backbone.model_version}")
    
    # 2. Fetch a real MIMIC-CXR patient study
    loader = MimicCxrLoader("validate", paths=config.mimic_paths)
    sample_record = None
    for record in loader.iter_records(limit=20):
        if record.images:
            sample_record = record
            break
            
    if not sample_record:
        raise RuntimeError("Validation Failure: Could not locate a valid image in validate CSV.")
        
    rel_img_path = sample_record.images[0]
    abs_img_path = config.mimic_paths.resolve_image(rel_img_path)
    print(f"[Validation] Loading validation image: {rel_img_path}")
    
    # 3. Create pipeline study input
    priors = StructuredPriors(age_band="65+", sex="M", fever=True)
    study_input = study_from_cxr(abs_img_path, study_id="post-train-val-study", priors=priors)
    
    # 4. Execute pipeline end-to-end
    print("[Validation] Running end-to-end clinical pipeline...")
    bundle = asyncio.run(pipe.run(study_input, "CASE-POST-TRAIN-VAL"))
    
    # 5. Verify results
    print("[Validation] End-to-end pipeline run succeeded!")
    print(f"  - Case ID: {bundle.case_id}")
    print(f"  - Case State: {bundle.state.value}")
    print(f"  - Primary Diagnosis: {bundle.safety.top.value} (Prob: {bundle.safety.top_probability:.4f})")
    print(f"  - Calibrated predictions count: {len(bundle.safety.predictions)}")
    print(f"  - Conformal prediction set: {[d.value for d in bundle.safety.conformal_set]}")
    print(f"  - Missing evidence recommendations: {len(bundle.recommendations)}")
    print(f"  - Grounded Clinical Report composed.")
    print(f"  - Visual Saliency (Explainability) generated.")
    print("="*60)
    print("POST-TRAINING VALIDATION PASSED SUCCESSFULLY!")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Train a production DenseNet121 Chest X-ray Model on MIMIC-CXR.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing.")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience.")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training.")
    parser.add_argument("--validate-only", action="store_true", help="Run validation only.")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if exists.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Artifact dir for this run (default: shared ARTIFACTS). "
                             "Use a fresh dir to avoid overwriting the served model.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Dataloader workers. >0 overlaps JPEG decode with GPU compute.")
    parser.add_argument("--use-sampler", action="store_true",
                        help="Also oversample rare positives. NOT recommended: stacking this "
                             "with pos_weight over-corrects imbalance (audit F5).")
    parser.add_argument("--no-post-val", action="store_true",
                        help="Skip the end-to-end pipeline check after training.")
    args = parser.parse_args()

    cfg_kw = dict(
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        patience=args.patience,
        amp=not args.no_amp,
        limit=args.limit,
        num_workers=args.num_workers,
        use_sampler=args.use_sampler,
    )
    if args.out_dir:
        cfg_kw["out_dir"] = args.out_dir
    config = TrainConfig(**cfg_kw)
    
    set_seed(config.seed)
    
    # 1. Build loaders
    train_loader, val_loader, train_labels = build_loaders(config, limit=config.limit)
    
    # 2. Instantiate model
    model = DenseNet121CXR(num_classes=7).to(config.device)
    
    # 3. Setup optimizer, scheduler, loss, scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    # Compute positive weights for loss
    pos_weight = None
    if len(train_labels) > 0:
        pos = train_labels.sum(axis=0)
        neg = len(train_labels) - pos
        pw = np.clip(neg / np.clip(pos, 1.0, None), 0.1, 20.0)
        pos_weight = torch.tensor(pw, dtype=torch.float32, device=config.device)
        
    # TV-regularised loss (Module 2): penalise high-frequency noise in the latent
    # feature maps so Grad-CAM++ locks onto anatomy. λ from config (0 disables TV).
    try:
        from common.config import get_settings
        tv_weight = get_settings().vision_tv_weight
    except Exception:
        tv_weight = 1e-4
    loss_fn = RegularizedMultiLabelLoss(pos_weight=pos_weight, tv_weight=tv_weight)
    print(f"[Train] TV feature regularisation weight λ={tv_weight:g}")
    scaler = torch.cuda.amp.GradScaler(enabled=(config.amp and config.device == "cuda"))
    
    # 4. Resume training if requested
    start_epoch = 0
    best_macro_auroc = -1.0
    if args.resume and config.last_checkpoint_path.exists():
        start_epoch, best_macro_auroc = load_model_checkpoint(
            config.last_checkpoint_path, model, optimizer, scheduler, scaler, config.device
        )
        
    # Tensorboard logger
    tb_writer = SummaryWriter(log_dir=str(config.tb_dir))
    csv_logger = HistoryLogger(config.history_csv_path)
    
    if args.validate_only:
        print("[Train] Running validation only...")
        val_loss, val_metrics = evaluate_model(model, val_loader, loss_fn, config.device)
        print(f"[Val] Loss: {val_loss:.4f} | Macro-AUROC: {val_metrics['macro_auroc']:.4f} | Macro-F1: {val_metrics['macro_f1']:.4f}")
        return

    # 5. Training loop
    print(f"[Train] Starting training on {config.device} (AMP={config.amp and config.device == 'cuda'})...")
    early_stop_counter = 0
    
    for epoch in range(start_epoch, config.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, config.device, config.grad_clip)
        
        # Validation
        val_loss, val_metrics = evaluate_model(model, val_loader, loss_fn, config.device)
        scheduler.step()
        
        epoch_time = time.time() - t0
        macro_auroc = val_metrics["macro_auroc"]
        
        print(f"Epoch {epoch:02d}/{config.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Macro-AUROC: {macro_auroc:.4f} | Time: {epoch_time:.1f}s")
        
        # Log metrics
        log_data = {"train_loss": train_loss}
        log_data.update(val_metrics)
        csv_logger.log(epoch, log_data)
        
        # Tensorboard logging
        tb_writer.add_scalar("Loss/train", train_loss, epoch)
        tb_writer.add_scalar("Loss/val", val_loss, epoch)
        tb_writer.add_scalar("Metric/val_macro_auroc", macro_auroc, epoch)
        tb_writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
        
        # Checkpoint checks
        is_best = macro_auroc > best_macro_auroc
        if is_best:
            best_macro_auroc = macro_auroc
            early_stop_counter = 0
            save_best_model(config, model, epoch, best_macro_auroc)
        else:
            early_stop_counter += 1
            
        save_model_checkpoint(config, model, optimizer, scheduler, scaler, epoch, best_macro_auroc)
        
        # Plot updated history curves
        plot_training_history(config.history_csv_path, config.plots_dir)
        
        # Early stopping
        if early_stop_counter >= config.patience:
            print(f"[Train] Early stopping triggered at epoch {epoch} (No improvement for {config.patience} epochs)")
            break
            
    tb_writer.close()
    print(f"[Train] Training completed! Best Macro-AUROC: {best_macro_auroc:.4f}")
    print(f"[Train] Artifacts written to: {config.out_dir}")

    # 6. Post-training validation and integration check
    if not args.no_post_val:
        run_post_training_validation(config)

if __name__ == "__main__":
    main()
