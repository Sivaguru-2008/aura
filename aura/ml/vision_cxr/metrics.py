import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

def compute_multilabel_metrics(probs, targets):
    """
    Computes classification metrics for multi-label chest X-ray predictions.
    
    probs: np.ndarray (N, num_classes) - raw probabilities
    targets: np.ndarray (N, num_classes) - binary ground truth (0 or 1)
    """
    num_classes = targets.shape[1]
    class_metrics = {}
    aucs = []
    f1s = []
    
    for i in range(num_classes):
        y_true = targets[:, i]
        y_prob = probs[:, i]
        y_pred = (y_prob >= 0.5).astype(float)
        
        # AUROC: require both negative and positive classes to be present in split
        if 0 < y_true.sum() < len(y_true):
            auc = roc_auc_score(y_true, y_prob)
            aucs.append(auc)
        else:
            auc = float('nan')
            
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        
        # Specificity = TN / (TN + FP)
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        spec = float(tn) / (tn + fp) if (tn + fp) > 0 else 0.0
        
        acc = (y_pred == y_true).mean()
        
        f1s.append(f1)
        
        class_metrics[f"class_{i}_auroc"] = float(auc)
        class_metrics[f"class_{i}_sensitivity"] = float(rec)  # Sensitivity is recall
        class_metrics[f"class_{i}_specificity"] = float(spec)
        class_metrics[f"class_{i}_f1"] = float(f1)
        class_metrics[f"class_{i}_accuracy"] = float(acc)
        
    class_metrics["macro_auroc"] = float(np.mean([a for a in aucs if not np.isnan(a)])) if aucs else 0.5
    class_metrics["macro_f1"] = float(np.mean(f1s)) if f1s else 0.0
    
    return class_metrics
