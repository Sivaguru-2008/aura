import torch
import numpy as np
from ml.vision_cxr.metrics import compute_multilabel_metrics
from ml.vision_cxr.losses import RegularizedMultiLabelLoss

def evaluate_model(model, dataloader, loss_fn, device):
    """Evaluates the model on the validation set, returning average loss and computed metrics."""
    model.eval()

    total_loss = 0.0
    all_probs = []
    all_targets = []
    # Report BCE-only validation loss: the TV term is a training regulariser and
    # needs a feature map that eval does not compute, so we score the data term.
    eval_loss = loss_fn.bce if isinstance(loss_fn, RegularizedMultiLabelLoss) else loss_fn

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(inputs)
            loss = eval_loss(logits, targets)
            
            total_loss += loss.item() * inputs.size(0)
            
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader.dataset)
    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    metrics = compute_multilabel_metrics(all_probs, all_targets)
    metrics["val_loss"] = avg_loss
    
    return avg_loss, metrics
