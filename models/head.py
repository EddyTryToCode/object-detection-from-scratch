"""
FCOS Detection Head (Anchor-free Object Detection).
Predicts for each pixel on the multi-scale feature maps:
  - cls_logits [C]: classification logits for each class
  - reg [4]: distance values (l, t, r, b) from the pixel to the 4 boundaries of the bounding box
  - centerness [1]: centerness score indicating closeness to the object center

Supports DCNv2 (Deformable Convolutional Network v2) for the last two convolutional layers of each tower.
When DCN is enabled, the layers learn offsets and masks dynamically to adapt to object shapes.
"""
import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d


class DeformBlock(nn.Module):
    """Deformable Convolution v2 block containing offset prediction, modulation mask, and DeformConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        k2 = kernel_size * kernel_size
        self.offset_conv = nn.Conv2d(in_channels, 2 * k2, kernel_size, padding=padding)
        self.mask_conv = nn.Conv2d(in_channels, k2, kernel_size, padding=padding)
        self.dcn = DeformConv2d(in_channels, out_channels, kernel_size, padding=padding)
        # Initialize weights to zero to start as standard convolution
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.zeros_(self.mask_conv.weight)
        nn.init.zeros_(self.mask_conv.bias)

    def forward(self, x):
        offset = self.offset_conv(x)
        mask = self.mask_conv(x).sigmoid()
        return self.dcn(x, offset, mask)


class FCOSHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=5, num_convs=4, use_dcn=False):
        super().__init__()
        self.num_classes = num_classes

        # Classification tower
        cls_layers = []
        for i in range(num_convs):
            if use_dcn and i >= num_convs - 2:
                cls_layers.extend([
                    DeformBlock(in_channels, in_channels),
                    nn.GroupNorm(32, in_channels),
                    nn.ReLU(inplace=True),
                ])
            else:
                cls_layers.extend([
                    nn.Conv2d(in_channels, in_channels, 3, padding=1),
                    nn.GroupNorm(32, in_channels),
                    nn.ReLU(inplace=True),
                ])
        self.cls_tower = nn.Sequential(*cls_layers)
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 3, padding=1)

        # Regression tower
        reg_layers = []
        for i in range(num_convs):
            if use_dcn and i >= num_convs - 2:
                reg_layers.extend([
                    DeformBlock(in_channels, in_channels),
                    nn.GroupNorm(32, in_channels),
                    nn.ReLU(inplace=True),
                ])
            else:
                reg_layers.extend([
                    nn.Conv2d(in_channels, in_channels, 3, padding=1),
                    nn.GroupNorm(32, in_channels),
                    nn.ReLU(inplace=True),
                ])
        self.reg_tower = nn.Sequential(*reg_layers)
        self.reg_pred = nn.Conv2d(in_channels, 4, 3, padding=1)

        # Centerness branch (shares features with regression tower)
        self.ctr_pred = nn.Conv2d(in_channels, 1, 3, padding=1)

        # Learnable scale parameters for regression per FPN level
        self.scales = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(3)])

        self._init_weights()

    def _init_weights(self):
        for m in [self.cls_tower, self.reg_tower]:
            for layer in m:
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    nn.init.zeros_(layer.bias)
        nn.init.constant_(self.cls_pred.bias, -2.0)

    def forward(self, x, scale_idx=0):
        cls_feat = self.cls_tower(x)
        reg_feat = self.reg_tower(x)

        cls_logits = self.cls_pred(cls_feat)                   # [B, C, H, W]
        reg_pred = torch.exp(self.reg_pred(reg_feat) * self.scales[scale_idx])  # [B, 4, H, W]
        ctr_pred = self.ctr_pred(reg_feat)                     # [B, 1, H, W]

        return cls_logits, reg_pred, ctr_pred
