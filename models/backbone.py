"""
EfficientNet-B3 Backbone Pretrained on ImageNet.

Outputs 3 multi-scale feature maps:
  - c3: stride 8,  channels = 48
  - c4: stride 16, channels = 136
  - c5: stride 32, channels = 1536
"""
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


class Backbone(nn.Module):
    # Feature channel dimensions utilized by neck and head modules
    C3_CHANNELS = 48
    C4_CHANNELS = 136
    C5_CHANNELS = 1536

    def __init__(self, pretrained=True):
        super().__init__()
        weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
        effnet = efficientnet_b3(weights=weights)
        self.features = effnet.features

    def forward(self, x):
        outputs = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 3:
                outputs['c3'] = x   # stride 8,  48 channels
            elif i == 5:
                outputs['c4'] = x   # stride 16, 136 channels
            elif i == 8:
                outputs['c5'] = x   # stride 32, 1536 channels
        return outputs
