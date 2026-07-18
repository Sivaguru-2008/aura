import torch
import torch.nn as nn
import torchvision.models as models

class DenseNet121CXR(nn.Module):
    """DenseNet-121 model adapted for grayscale chest X-rays and multi-label pathology classification."""
    def __init__(self, num_classes=7):
        super().__init__()
        
        # Load DenseNet121 with ImageNet weights, handling API differences across torchvision versions
        try:
            from torchvision.models import densenet121, DenseNet121_Weights
            self.densenet = densenet121(weights=DenseNet121_Weights.DEFAULT)
        except ImportError:
            from torchvision.models import densenet121
            self.densenet = densenet121(pretrained=True)
            
        # Modify the first conv layer to accept grayscale (1 channel) instead of RGB (3 channels)
        # Original: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        original_conv = self.densenet.features.conv0
        self.densenet.features.conv0 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )
        
        # Average the weights of the original 3 channels to initialize the new 1-channel conv layer
        with torch.no_grad():
            self.densenet.features.conv0.weight.copy_(
                original_conv.weight.sum(dim=1, keepdim=True)
            )
            
        # Modify the classification head to map to our 7 findings classes
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.densenet(x)
