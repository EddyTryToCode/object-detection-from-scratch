"""
Unified Detector Factory (Configured specifically for FCOS architecture).
"""
import torch
import torch.nn as nn
from models.backbone import Backbone


class Detector(nn.Module):
    """Detector model class for FCOS (Fully Convolutional One-Stage Object Detection)."""

    def __init__(self, num_classes=5, pretrained_backbone=True,
                 use_sppf=True, attention='cbam', use_dcn=True):
        super().__init__()
        self.arch = 'fcos'
        self.backbone = Backbone(pretrained=pretrained_backbone)

        from models.neck import FPN
        from models.head import FCOSHead
        
        in_ch = [self.backbone.C3_CHANNELS, self.backbone.C4_CHANNELS, self.backbone.C5_CHANNELS]

        # Optional: SPPF on C5 (receptive field expansion)
        if use_sppf:
            from models.attention import SPPF
            self.sppf = SPPF(in_ch[2], in_ch[2])

        # Optional: Attention blocks on feature maps
        if attention == 'cbam':
            from models.attention import CBAM
            self.attn_c3 = CBAM(in_ch[0])
            self.attn_c4 = CBAM(in_ch[1])
            self.attn_c5 = CBAM(in_ch[2])
        elif attention == 'eca':
            from models.attention import ECABlock
            self.attn_c3 = ECABlock(in_ch[0])
            self.attn_c4 = ECABlock(in_ch[1])
            self.attn_c5 = ECABlock(in_ch[2])

        self.neck = FPN(in_channels_list=in_ch, out_channels=256)
        self.head = FCOSHead(in_channels=256, num_classes=num_classes, use_dcn=use_dcn)

    def forward(self, x):
        features = self.backbone(x)

        # Apply SPPF on C5 if configured
        if hasattr(self, 'sppf'):
            features['c5'] = self.sppf(features['c5'])
        # Apply attention on c3/c4/c5 if configured
        if hasattr(self, 'attn_c3'):
            features['c3'] = self.attn_c3(features['c3'])
            features['c4'] = self.attn_c4(features['c4'])
            features['c5'] = self.attn_c5(features['c5'])

        fpn_out = self.neck(features)
        keys = ['p3', 'p4', 'p5']
        return [self.head(fpn_out[k], scale_idx=i) for i, k in enumerate(keys)]
