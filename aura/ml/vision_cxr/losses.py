import torch
import torch.nn as nn


class MultiLabelLoss(nn.Module):
    """Multi-label binary cross entropy loss with logits, supporting class weights."""
    def __init__(self, pos_weight=None):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)


class TotalVariationLoss(nn.Module):
    r"""Total-Variation penalty on latent feature maps (Module 2 feature regulariser).

    Motivation
    ----------
    Grad-CAM++ differentiates through the last conv feature map. If that map carries
    high-frequency, spatially-incoherent activation noise, the resulting heatmap is
    speckled and does not lock onto anatomy. Penalising the **total variation** of
    the feature map pushes activations to be piecewise-smooth — flat over a region,
    changing sharply only at true structural boundaries — which is exactly the prior
    that makes attributions land on pathology edges rather than texture noise.

    Definition (anisotropic ℓ1 TV, averaged over channels & batch):

        TV(F) = Σ_{b,k} [ Σ_{i,j} |F[i+1,j] − F[i,j]| + |F[i,j+1] − F[i,j]| ] / Z

    with ``Z`` the element count so the scale is resolution-independent. The
    isotropic variant ``Σ sqrt(dx² + dy² + ε)`` is available via ``isotropic=True``
    (rotationally invariant, slightly stronger edge preservation).

    Caveat (documented, not hidden): TV is a smoothness prior. Too large a weight
    over-smooths and can erase genuinely small, high-frequency lesions — nodules and
    thin pneumothorax lines. Keep ``λ`` small (config default ``1e-4``) and treat it
    as a tunable, validated against small-lesion recall, not a free win.
    """
    def __init__(self, weight: float = 1e-4, isotropic: bool = False, eps: float = 1e-6):
        super().__init__()
        self.weight = float(weight)
        self.isotropic = bool(isotropic)
        self.eps = float(eps)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """``feature_map``: (B, C, H, W). Returns the weighted scalar TV loss."""
        f = feature_map
        dh = f[:, :, 1:, :] - f[:, :, :-1, :]      # vertical differences
        dw = f[:, :, :, 1:] - f[:, :, :, :-1]      # horizontal differences
        if self.isotropic:
            # Align shapes on the overlapping interior and combine as a gradient magnitude.
            dh2 = dh[:, :, :, :-1] ** 2
            dw2 = dw[:, :, :-1, :] ** 2
            tv = torch.sqrt(dh2 + dw2 + self.eps).mean()
        else:
            tv = dh.abs().mean() + dw.abs().mean()
        return self.weight * tv


class RegularizedMultiLabelLoss(nn.Module):
    """Convenience wrapper: BCE-with-logits + TV(latent features) in one call.

    Usage in the training loop:
        logits, feats = model(inputs, return_features=True)
        loss, parts = criterion(logits, targets, feats)
    ``parts`` breaks out {"bce", "tv"} for logging.
    """
    def __init__(self, pos_weight=None, tv_weight: float = 1e-4, tv_isotropic: bool = False):
        super().__init__()
        self.bce = MultiLabelLoss(pos_weight=pos_weight)
        self.tv = TotalVariationLoss(weight=tv_weight, isotropic=tv_isotropic)

    def forward(self, logits, targets, feature_map=None):
        bce = self.bce(logits, targets)
        tv = self.tv(feature_map) if (feature_map is not None and self.tv.weight > 0) else logits.new_zeros(())
        total = bce + tv
        return total, {"bce": float(bce.detach()), "tv": float(tv.detach()) if torch.is_tensor(tv) else 0.0}
