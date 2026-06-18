"""
Path Aggregation Feature Pyramid Network (PAFPN) Neck.
Combines Feature Pyramid Network (FPN) with Bottom-Up Path Aggregation (similar to PANet).
The bottom-up pathway facilitates localization feature propagation from lower spatial layers to higher semantic layers.
"""
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """PAFPN: FPN with additional Bottom-Up path aggregation."""

    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        c3_ch, c4_ch, c5_ch = in_channels_list

        # Lateral 1x1 convolutions to project feature maps into uniform channel size
        self.lat_c5 = nn.Conv2d(c5_ch, out_channels, 1)
        self.lat_c4 = nn.Conv2d(c4_ch, out_channels, 1)
        self.lat_c3 = nn.Conv2d(c3_ch, out_channels, 1)

        # Smooth convolutions for top-down feature maps
        self.td_p5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.td_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.td_p3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Bottom-up path aggregation blocks (stride=2 downsampling + smooth conv)
        self.bu_down_3to4 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.bu_smooth_4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bu_down_4to5 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.bu_smooth_5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, features):
        c3, c4, c5 = features['c3'], features['c4'], features['c5']

        # Top-down pathway construction
        p5 = self.lat_c5(c5)
        p4 = self.lat_c4(c4) + F.interpolate(p5, size=c4.shape[2:], mode='nearest')
        p3 = self.lat_c3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='nearest')

        p5 = self.td_p5(p5)
        p4 = self.td_p4(p4)
        p3 = self.td_p3(p3)

        # Bottom-up pathway construction
        n3 = p3
        n4 = self.bu_smooth_4(p4 + self.bu_down_3to4(n3))
        n5 = self.bu_smooth_5(p5 + self.bu_down_4to5(n4))

        return {'p3': n3, 'p4': n4, 'p5': n5}
