import torch
import torch.nn as nn

class MultiLabelLoss(nn.Module):
    """Multi-label binary cross entropy loss with logits, supporting class weights."""
    def __init__(self, pos_weight=None):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)
