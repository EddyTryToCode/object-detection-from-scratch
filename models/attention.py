"""
Attention & Pooling Modules for Object Detection.
All modules are designed to be plug-and-play, inserted between the backbone and neck.

Modules:
    - CBAM: Convolutional Block Attention Module (Channel + Spatial Attention).
    - ECABlock: Efficient Channel Attention (lightweight 1D convolution).
    - SPPF: Spatial Pyramid Pooling Fast (expands C5 feature map receptive field).
"""
import math
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────
#  CBAM: Convolutional Block Attention Module
# ─────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """CBAM Channel Attention: evaluates channel-wise feature importance."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x):
        B, C = x.shape[:2]
        avg_out = self.mlp(x.mean(dim=[2, 3]))        # [B, C]
        max_out = self.mlp(x.amax(dim=[2, 3]))        # [B, C]
        scale = (avg_out + max_out).sigmoid().view(B, C, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    """CBAM Spatial Attention: evaluates spatial location importance."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)          # [B, 1, H, W]
        max_out = x.amax(dim=1, keepdim=True)           # [B, 1, H, W]
        scale = self.conv(torch.cat([avg_out, max_out], dim=1)).sigmoid()
        return x * scale


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel → spatial)."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


# ─────────────────────────────────────────────────────────────────────
#  ECA: Efficient Channel Attention
# ─────────────────────────────────────────────────────────────────────

class ECABlock(nn.Module):
    """Efficient Channel Attention: uses 1D convolution for extremely lightweight channel-wise attention."""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # Adaptive kernel size dựa trên channel count
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        k = max(k, 3)  # Minimum kernel size of 3
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x):
        y = self.avg_pool(x)                               # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)                # [B, 1, C]
        y = self.conv(y)                                   # [B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)              # [B, C, 1, 1]
        return x * y.sigmoid()


# ─────────────────────────────────────────────────────────────────────
#  SPPF: Spatial Pyramid Pooling - Fast (YOLOv5)
# ─────────────────────────────────────────────────────────────────────

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast.
    Applies three sequential MaxPool operations, concatenates their outputs, and projects with a 1x1 Conv.
    Significantly expands the receptive field of the C5 feature map.
    """
    def __init__(self, in_channels, out_channels, k=5):
        super().__init__()
        hidden = in_channels // 2
        self.cv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(k, stride=1, padding=k // 2)
        self.cv2 = nn.Sequential(
            nn.Conv2d(hidden * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))
