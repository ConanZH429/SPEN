import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from timm.layers.mlp import Mlp, GluMlp
from timm.layers.conv_bn_act import ConvNormAct
from timm.models._efficientnet_blocks import UniversalInvertedResidual

from timm.layers.squeeze_excite import SEModule
from timm.layers.cbam import SpatialAttn, CbamModule

from typing import Optional


ConvAct = nn.SiLU
MLPAct = nn.SiLU


class DensFuse(nn.Module):
    def __init__(self, align_channels: int):
        super().__init__()
        self.conv_downsample = ConvNormAct(align_channels, align_channels, 3, stride=2, act_layer=ConvAct)
    
    def forward(self, shallow_feature: Tensor, current_feature: Tensor, deep_feature: Tensor):
        feature_fused = self.conv_downsample(shallow_feature) + current_feature + F.interpolate(deep_feature, scale_factor=2, mode='bilinear', align_corners=False)
        return feature_fused


class SSIAFuse(nn.Module):
    def __init__(self, align_channels: int):
        super().__init__()
        # spatial
        self.conv3x3 = ConvNormAct(align_channels, 1, 3, stride=2, dilation=1, act_layer=ConvAct)
        self.conv5x5 = ConvNormAct(align_channels, 1, 5, stride=2, dilation=1, act_layer=ConvAct)
        self.conv7x7 = ConvNormAct(align_channels, 1, 3, stride=2, dilation=3, act_layer=ConvAct)
        self.conv9x9 = ConvNormAct(align_channels, 1, 5, stride=2, dilation=2, act_layer=ConvAct)
        self.spatial_weight_conv = ConvNormAct(4, 1, 3, apply_act=False, apply_norm=False)

        # channel
        self.feature_weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.channel_weight_conv = nn.Sequential(
            nn.Conv2d(align_channels, align_channels // 2, 1, bias=False),
            ConvAct(inplace=True),
            nn.Conv2d(align_channels // 2, align_channels, 1, bias=False),
        )

        # fuse
        self.conv_downsample = ConvNormAct(align_channels, align_channels, 3, stride=2, act_layer=ConvAct)
    
    def forward(self, shallow_feature: Tensor, current_feature: Tensor, deep_feature: Tensor):
        # spatial
        spatial_feature = torch.cat([
            self.conv3x3(shallow_feature),
            self.conv5x5(shallow_feature),
            self.conv7x7(shallow_feature),
            self.conv9x9(shallow_feature),
        ], dim=1)
        spatial_weight = F.sigmoid(self.spatial_weight_conv(spatial_feature))  # B, 1, H, W

        # channel
        avg_feature = F.adaptive_avg_pool2d(deep_feature, 1)
        max_feature = F.adaptive_max_pool2d(deep_feature, 1)
        weight = F.sigmoid(self.feature_weight)
        channel_feature = avg_feature * weight[0] + max_feature * weight[1]
        channel_weight = F.sigmoid(self.channel_weight_conv(channel_feature))  # B, C, 1, 1

        # fusion
        feature_fused = self.conv_downsample(shallow_feature) + current_feature + F.interpolate(deep_feature, scale_factor=2, mode='bilinear', align_corners=False)
        feature_fused = feature_fused * spatial_weight * channel_weight
        return feature_fused


class Fuse(nn.Module):
    def __init__(self, align_channels: int):
        super().__init__()
        self.conv_downsample = ConvNormAct(align_channels, align_channels, 3, stride=2, act_layer=ConvAct)
        self.attention = nn.Identity()
    
    def forward(self, shallow_feature: Tensor, current_feature: Tensor, deep_feature: Tensor):
        feature_fused = self.conv_downsample(shallow_feature) + current_feature + F.interpolate(deep_feature, scale_factor=2, mode='bilinear', align_corners=False)
        feature_fused = self.attention(feature_fused)
        return feature_fused


class SEFuse(Fuse):
    def __init__(self, align_channels: int):
        super().__init__(align_channels)
        self.attention = SEModule(align_channels, rd_ratio=1./16, act_layer=ConvAct)


class SAMFuse(Fuse):
    def __init__(self, align_channels: int):
        super().__init__(align_channels)
        self.attention = SpatialAttn()

class CBAMFuse(Fuse):
    def __init__(self, align_channels: int):
        super().__init__(align_channels)
        self.attention = CbamModule(align_channels, rd_ratio=1./16, act_layer=ConvAct)


class AttFuse(nn.Module):
    def __init__(self, align_channels: int, att_type: Optional[str] = None):
        super().__init__()
        if att_type == "SE":
            self.attention = SEFuse(align_channels)
        elif att_type == "SAM":
            self.attention = SAMFuse(align_channels)
        elif att_type == "CBAM":
            self.attention = CBAMFuse(align_channels)
        elif att_type == "SSIA":
            self.attention = SSIAFuse(align_channels)
        elif att_type is None:
            self.attention = Fuse(align_channels)
        else:
            raise ValueError(f"Unknown attention type: {att_type}")
    
    def forward(self, shallow_feature: Tensor, current_feature: Tensor, deep_feature: Tensor):
        return self.attention(shallow_feature, current_feature, deep_feature)