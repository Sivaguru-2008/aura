import torch
import torch.nn as nn
import torchvision.models as models

# ITU-R BT.601 luma coefficients (R, G, B). These are the weights that map an RGB
# image to perceived luminance — the channel a grayscale radiograph corresponds to.
LUMA_BT601 = (0.299, 0.587, 0.114)


def luminance_init_conv0(original_conv: nn.Conv2d,
                         coeffs=LUMA_BT601) -> nn.Conv2d:
    """Build a 1-channel conv0 that *conserves* the pretrained ImageNet filters.

    Why weighted, not summed or duplicated
    --------------------------------------
    The original conv0 sees a 3-channel input. Present a grayscale image ``g`` to it
    the naive way — broadcast ``g`` to all three channels — and each output filter
    ``o`` responds with

        y_o = Σ_c W[o, c] * g = ( Σ_c W[o, c] ) * g,

    i.e. the **plain channel sum** ``Σ_c W[o,c]`` (this is what the previous code
    used: ``original_conv.weight.sum(dim=1)``). That treats R, G, B as equally
    relevant to luminance, which they are not: it inflates the filter gain by
    treating three correlated channels as independent, shifting the activation
    statistics away from what BatchNorm downstream was calibrated to.

    Instead we collapse the three channels with the **luminance weights** the
    network was implicitly trained under:

        W_new[o, 0] = α·W[o,R] + β·W[o,G] + γ·W[o,B],   (α,β,γ) = BT.601.

    This reconstructs each filter's response to the *luminance* component of a
    colour image — precisely the signal a grayscale CXR carries — so the edge- and
    texture-detectors keep their tuned orientation and scale instead of being
    diluted. The result: sharper deep features and better-grounded Grad-CAM++
    (feature dilution is what produced the noisy, ungrounded heatmaps).
    """
    a, b, c = coeffs
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=(original_conv.bias is not None),   # was `bias=original_conv.bias` (a Tensor/None)
    )
    with torch.no_grad():
        w = original_conv.weight                 # (out, 3, kH, kW)
        weighted = a * w[:, 0:1] + b * w[:, 1:2] + c * w[:, 2:3]   # (out, 1, kH, kW)
        new_conv.weight.copy_(weighted)
        if original_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(original_conv.bias)
    return new_conv


class DenseNet121CXR(nn.Module):
    """DenseNet-121 adapted for grayscale chest X-rays and multi-label pathology.

    ``return_features=True`` on forward also returns the pre-pool feature map, which
    the trainer feeds to the Total-Variation regulariser (see ``losses.py``).
    """
    def __init__(self, num_classes=7, luma_coeffs=LUMA_BT601):
        super().__init__()

        # Load DenseNet121 with ImageNet weights, handling API differences across versions.
        try:
            from torchvision.models import densenet121, DenseNet121_Weights
            self.densenet = densenet121(weights=DenseNet121_Weights.DEFAULT)
        except ImportError:
            from torchvision.models import densenet121
            self.densenet = densenet121(pretrained=True)

        # Grayscale conv0 via luminance-weighted summation of the pretrained RGB filters.
        self.densenet.features.conv0 = luminance_init_conv0(
            self.densenet.features.conv0, coeffs=luma_coeffs
        )

        # Classification head over our finding set.
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x, return_features: bool = False):
        # Mirror torchvision DenseNet.forward but expose the latent feature map.
        features = self.densenet.features(x)                     # (B, 1024, 7, 7)
        out = torch.relu(features)
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        logits = self.densenet.classifier(out)
        if return_features:
            return logits, features
        return logits
