import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from timm.layers.mlp import Mlp, GluMlp
from timm.layers.conv_bn_act import ConvNormAct
from timm.models._efficientnet_blocks import UniversalInvertedResidual
from timm.models.vision_transformer import Block, Attention

from timm.layers.squeeze_excite import SEModule
from timm.layers.cbam import SpatialAttn, CbamModule

from timm.layers.weight_init import trunc_normal_

from typing import Optional, List, Tuple, Union, Literal


ConvAct = nn.Mish
MLPAct = nn.Mish


class DensFuse(nn.Module):
    def __init__(self, align_channels: int):
        super().__init__()
        self.conv_downsample = ConvNormAct(align_channels, align_channels, 3, stride=2, act_layer=ConvAct)
    
    def forward(self, shallow_feature: Tensor, current_feature: Tensor, deep_feature: Tensor):
        feature_fused = self.conv_downsample(shallow_feature) + current_feature + F.interpolate(deep_feature, size=current_feature[-2:], mode='bilinear', align_corners=True)
        return feature_fused


class SSIAFuse(nn.Module):
    def __init__(self, in_channels: List[int]):
        super().__init__()
        # spatial
        self.conv3x3 = ConvNormAct(in_channels[0], 1, 3, stride=1, dilation=1, act_layer=ConvAct)
        self.conv5x5 = ConvNormAct(in_channels[0], 1, 5, stride=1, dilation=1, act_layer=ConvAct)
        self.spatial_weight_conv = ConvNormAct(2, 1, 3, stride=2, apply_act=False, apply_norm=False)

        # channel
        self.feature_weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.channel_weight_conv = nn.Sequential(
            nn.Conv2d(in_channels[-1], in_channels[-1], 1, bias=False),
            ConvAct(inplace=True),
            nn.Conv2d(in_channels[-1], sum(in_channels) , 1, bias=False),
        )

        # fuse
        self.conv_downsample = ConvNormAct(in_channels[0], in_channels[0], 3, stride=2, act_layer=ConvAct)
    
    def forward(self, shallow_feature: Tensor, current_feature: Tensor, deep_feature: Tensor):
        # spatial
        spatial_feature = torch.cat([
            self.conv3x3(shallow_feature),
            self.conv5x5(shallow_feature),
        ], dim=1)
        spatial_weight = F.sigmoid(self.spatial_weight_conv(spatial_feature))  # B, 1, H, W

        # channel
        avg_feature = F.adaptive_avg_pool2d(deep_feature, 1)
        max_feature = F.adaptive_max_pool2d(deep_feature, 1)
        weight = F.sigmoid(self.feature_weight)
        channel_feature = avg_feature * weight[0] + max_feature * weight[1]
        channel_weight = F.sigmoid(self.channel_weight_conv(channel_feature))  # B, C, 1, 1

        # fusion
        # feature_fused = self.conv_downsample(shallow_feature) + current_feature + F.interpolate(deep_feature, size=current_feature.shape[-2:], mode='bilinear', align_corners=True)
        feature_fused = torch.cat([
            self.conv_downsample(shallow_feature),
            current_feature,
            F.interpolate(deep_feature, size=current_feature.shape[-2:], mode='bilinear', align_corners=True)
        ], dim=1)
        feature_fused = feature_fused * spatial_weight * channel_weight
        return feature_fused


class Fuse(nn.Module):
    def __init__(self, in_channels: List[int]):
        super().__init__()
        self.conv_downsample = ConvNormAct(in_channels[0], in_channels[0], 3, stride=2, act_layer=ConvAct)
        self.attention = nn.Identity()
    
    def forward(self, shallow_feature: Tensor, current_feature: Tensor, deep_feature: Tensor):
        feature_fused = torch.cat([
            self.conv_downsample(shallow_feature),
            current_feature,
            F.interpolate(deep_feature, size=current_feature.shape[-2:], mode='bilinear', align_corners=True)
        ], dim=1)
        feature_fused = self.attention(feature_fused)
        return feature_fused


class SEFuse(Fuse):
    def __init__(self, in_channels: List[int]):
        super().__init__(in_channels)
        self.attention = SEModule(sum(in_channels), rd_ratio=1./16, act_layer=ConvAct)


class SAMFuse(Fuse):
    def __init__(self, in_channels: List[int]):
        super().__init__(in_channels)
        self.attention = SpatialAttn()

class CBAMFuse(Fuse):
    def __init__(self, in_channels: List[int]):
        super().__init__(in_channels)
        self.attention = CbamModule(sum(in_channels), rd_ratio=1./8, act_layer=ConvAct)


class AttFuse(nn.Module):
    def __init__(self, in_channels: List[int], att_type: Optional[str] = None):
        super().__init__()
        if att_type == "SE":
            self.attention = SEFuse(in_channels)
        elif att_type == "SAM":
            self.attention = SAMFuse(in_channels)
        elif att_type == "CBAM":
            self.attention = CBAMFuse(in_channels)
        elif att_type == "SSIA":
            self.attention = SSIAFuse(in_channels)
        elif att_type is None:
            self.attention = Fuse(in_channels)
        else:
            raise ValueError(f"Unknown attention type: {att_type}")
    
    def forward(self, shallow_feature: Tensor, current_feature: Tensor, deep_feature: Tensor):
        return self.attention(shallow_feature, current_feature, deep_feature)


class AvgPool(nn.Module):
    def __init__(self, pool_size: int):
        super().__init__()
        self.avg_size = pool_size
    
    def forward(self, x: Tensor):
        return F.adaptive_avg_pool2d(x, self.pool_size).flatten(1)      # B, C, H, W -> B, C*H*W


class MaxPool(nn.Module):
    def __init__(self, pool_size: int):
        super().__init__()
        self.pool_size = pool_size
    
    def forward(self, x: Tensor):
        return F.adaptive_max_pool2d(x, self.pool_size).flatten(1)      # B, C, H, W -> B, C*H*W


class MixPool(nn.Module):
    def __init__(self, pool_size: int, weighted_learnable: bool = False):
        super().__init__()
        self.avg_pool = AvgPool(pool_size)
        self.max_pool = MaxPool(pool_size)
        if weighted_learnable:
            self.weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        else:
            self.register_buffer("weight", torch.tensor([1, 1], dtype=torch.float32, requires_grad=False))
    
    def forward(self, x: Tensor):
        super().__init__()
        avg_feature = self.avg_pool(x)  # B, C, H, W -> B, C*H*W
        max_feature = self.max_pool(x)  # B, C, H, W -> B, C*H*W
        weight = F.sigmoid(self.weight) # B, 2
        feature = avg_feature * weight[0] + max_feature * weight[1] # B, C*H*W
        return feature  # B, C*H*W


class MHA(nn.Module):
    def __init__(self, in_channels: int, patched_shape: Tuple[int, int], num_heads: int = 8, key_ratio: float = 0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.key_dim = int(in_channels * key_ratio)
        self.patched_shape = patched_shape
        self.scale = self.head_dim ** -0.5
        self.key_ratio = key_ratio
        nhxkd = num_heads * self.key_dim
        self.qkv = nn.Linear(in_channels, )
    
    def forward(self, x: Tensor):
        pass


class AvgPool(nn.Module):
    def __init__(self,
                 pool_size: Union[int, Tuple[int, int]],
                 single: bool = False):
        super().__init__()
        self.pool_size = pool_size
        if single:
            self.forward = self.forward_single
    
    def forward(self, x: Tensor):
        pos_feature = F.adaptive_avg_pool2d(x, self.pool_size).flatten(2)      # B, C, H, W -> B, C, S
        ori_feature = pos_feature
        return pos_feature, ori_feature

    def forward_single(self, x: Tensor):
        return F.adaptive_avg_pool2d(x, self.pool_size).flatten(2)      # B, C, H, W -> B, C, S


class MaxPool(nn.Module):
    def __init__(self,
                 pool_size: Union[int, Tuple[int, int]],
                 single: bool = False):
        super().__init__()
        self.pool_size = pool_size
        if single:
            self.forward = self.forward_single
    
    def forward(self, x: Tensor):
        pos_feature = F.adaptive_max_pool2d(x, self.pool_size).flatten(2)      # B, C, H, W -> B, C, S
        ori_feature = pos_feature
        return pos_feature, ori_feature

    def forward_single(self, x: Tensor):
        return F.adaptive_max_pool2d(x, self.pool_size).flatten(2)      # B, C, H, W -> B, C, S


class MixPool(nn.Module):
    def __init__(self,
                 pool_size: Union[int, Tuple[int, int]],
                 weighted_learnable: bool = False,
                 single: bool = False):
        super().__init__()
        self.avg_pool = AvgPool(pool_size, single=True)
        self.max_pool = MaxPool(pool_size, single=True)
        if single:
            self.forward = self.forward_single
            if weighted_learnable:
                self.weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            else:
                self.register_buffer("weight", torch.tensor([1, 1], dtype=torch.float32, requires_grad=False))
        else:
            if weighted_learnable:
                self.weight = nn.Parameter(torch.ones(4, dtype=torch.float32), requires_grad=True)
            else:
                self.register_buffer("weight", torch.tensor([1, 1, 1, 1], dtype=torch.float32, requires_grad=False))
    
    def forward(self, x: Tensor):
        avg_feature = self.avg_pool(x)  # B, C, H, W -> B, C, S
        max_feature = self.max_pool(x)  # B, C, H, W -> B, C, S
        weight = F.sigmoid(self.weight) # 4
        pos_feature = avg_feature * weight[0] + max_feature * weight[1] # B, C, S
        ori_feature = avg_feature * weight[2] + max_feature * weight[3] # B, C, S
        return pos_feature, ori_feature  # B, C, S

    def forward_single(self, x: Tensor):
        avg_feature = self.avg_pool(x)
        max_feature = self.max_pool(x)
        weight = F.sigmoid(self.weight)
        return avg_feature * weight[0] + max_feature * weight[1]


class SPP(nn.Module):
    def __init__(self,
                 pool_size: Union[Tuple[int], Tuple[Tuple[int, int]]] = (1, 2),
                 mode:  Literal["max", "mean", "mix"] = "max",
                 single: bool = False):
        super().__init__()
        self.pool_size = pool_size
        self.mode = mode

        if self.mode == "max":
            self.pool_list = nn.ModuleList([MaxPool(size, single=single) for size in self.pool_size])
        elif self.mode == "mean":
            self.pool_list = nn.ModuleList([AvgPool(size, single=single) for size in self.pool_size])
        elif self.mode == "mix":
            self.pool_list = nn.ModuleList([MixPool(size, single=single) for size in self.pool_size])
        else:
            raise ValueError(f"Unknown spp pooling mode: {mode}")
        
        if single:
            self.forward = self.forward_single
    
    def forward(self, x: Tensor):
        pooled_feature = [pool(x) for pool in self.pool_list]           # B, C, H, W -> B, C, S
        pos_feature = torch.cat([feature[0] for feature in pooled_feature], dim=-1)
        ori_feature = torch.cat([feature[1] for feature in pooled_feature], dim=-1)
        return pos_feature, ori_feature

    def forward_single(self, x: Tensor):
        return torch.cat([pool(x) for pool in self.pool_list], dim=-1)


class PatchEmbedding(nn.Module):
    def __init__(self,
                 patch_size: Optional[Union[int, Tuple[int, int]]],
                 embedding_mode: Literal["mean", "max", "mix"] = "max"):
        super().__init__()
        self.patch_size = patch_size
        
        if patch_size is None:
            self.patch_embedding = nn.Flatten(2)
        elif embedding_mode == "mean":
            self.patch_embedding = AvgPool(patch_size, single=True)
        elif embedding_mode == "max":
            self.patch_embedding = MaxPool(patch_size, single=True)
        elif embedding_mode == "mix":
            self.patch_embedding = MixPool(patch_size, single=True)
        else:
            raise ValueError(f"Unknown embedding mode: {embedding_mode}")
    
    def forward(self, x: Tensor):
        return self.patch_embedding(x).transpose(1, 2)  # B, C, H, W -> B, C, S -> B, S, C


class MHAPool(nn.Module):
    def __init__(self,
                 in_channels: int,
                 patch_size: Optional[Union[int, Tuple[int, int]]],
                 embedding_mode: Literal["max", "mean", "mix"],
                 pool_size: Union[int, Tuple[int, int], Tuple[Tuple[int, int]]],
                 pool_mode: Literal["max", "mean", "mix", "sppmax", "sppmean", "sppmix"] = "sppmax",
                 num_heads: int = 8,
                 single: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.num_heads = num_heads

        self.patch_embedding = PatchEmbedding(patch_size, embedding_mode)

        self.att = Attention(
            dim=in_channels,
            num_heads=num_heads,
            qkv_bias=True,
        )

        if pool_mode == "max":
            self.pool = MaxPool(pool_size, single=single)
        elif pool_mode == "mean":
            self.pool = AvgPool(pool_size, single=single)
        elif pool_mode == "mix":
            self.pool = MixPool(pool_size, single=single)
        elif pool_mode == "sppmax":
            self.pool = SPP(pool_size, mode="max", single=single)
        elif pool_mode == "sppmean":
            self.pool = SPP(pool_size, mode="mean", single=single)
        elif pool_mode == "sppmix":
            self.pool = SPP(pool_size, mode="mix", single=single)
        else:
            raise ValueError(f"Unknown pooling mode: {pool_mode}")
    
    def forward(self, x: Tensor):
        B, C, H, W = x.shape
        patch = self.patch_embedding(x)                       # B, C, H, W -> B, S, C
        patch = self.att(patch)                                 # B, S, C
        feature = patch.transpose(1, 2).reshape(B, C, H, W)    # B, S, C -> B, C, S -> B, C, H, W
        return self.pool(feature)                               # B, C, H, W -> B, C, S


class TokenFeature(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 patch_size: Optional[Union[int, Tuple[int, int]]],
                 embedding_mode: Literal["max", "mean", "mix", "sppmax", "sppmean", "sppmix"] = "max",
                 num_heads: int = 8,
                 num_layers: int = 5,
                 learnable_token_num: int = 2,
                 single: bool = False,):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads

        self.patch_embedding = PatchEmbedding(patch_size, embedding_mode)

        self.blocks = nn.Sequential(*[
            Block(
                dim=in_channels,
                num_heads=num_heads,
            ) for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(in_channels)
        
        if single:
            self.token_embedding = nn.Embedding(1, in_channels)
            self.register_buffer("token", torch.tensor([0], dtype=torch.long, requires_grad=False))
            self.add_learnable_token = self.add_single_learnable_token
            self.forward = self.forward_single
        else:
            self.learnable_token_num = learnable_token_num
            self.token_embedding = nn.Embedding(self.learnable_token_num, in_channels)
            self.register_buffer("token", torch.tensor([i for i in range(self.learnable_token_num)], dtype=torch.long, requires_grad=False))
        
        self.position_embedding = nn.Embedding(1000, in_channels)
        self.weight_init()
    
    def add_learnable_token(self, B:int, patch: Tensor):
        token = self.token.repeat(B, 1)
        pos_ori_token_embedded = self.token_embedding(token)
        token_embedded = torch.cat([pos_ori_token_embedded, patch], dim=1)
        position_token = torch.arange(0, patch.size(1)+self.learnable_token_num, device=patch.device)                    # S+2
        position_embedded = self.position_embedding(position_token)        # S+2, C
        return token_embedded + position_embedded

    def add_single_learnable_token(self, B:int, patch: Tensor):
        token = self.token.expand(B, 1)
        token_embedded = self.token_embedding(token)                # B, learnable_token, embedding_dim
        token_embedded = torch.cat([token_embedded, patch], dim=1)
        position_token = torch.arange(0, patch.size(1)+self.learnable_token_num, device=patch.device)                    # S+2
        position_embedded = self.position_embedding(position_token)
        return token_embedded + position_embedded.unsqueeze(dim=0)

    def forward(self, x: Tensor):
        B, C, H, W = x.shape
        patch = self.patch_embedding(x)       # B, C, H, W -> B, S, C
        patch = self.add_learnable_token(B, patch)
        out = self.blocks(patch)                     # B, 2+S, C -> B, 2+S, C
        out = self.layer_norm(out)
        out = out.transpose(1, 2)                   # B, 2+S, C -> B, C, 2+S
        if self.learnable_token_num == 6:
            return out[:, :, :6]
        else:
            return out[:, :, 0], out[:, :, 1]

    def forward_single(self, x: Tensor):
        B, C, H, W = x.shape
        patch = self.patch_embedding(x)       # B, C, H, W -> B, C, S -> B, S, C
        patch = self.add_learnable_token(B, patch)
        out = self.blocks(patch)                     # B, 1+S, C -> B, 1+S, C
        out = out.transpose(1, 2)                   # B, 1+S, C -> B, C, 1+S
        return out[:, :, 0:1]                        # B, C, 1
    
    def weight_init(self):
        trunc_normal_(self.position_embedding.weight, std=.02)
        nn.init.normal_(self.token_embedding.weight)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif hasattr(m, "init_weights"):
                m.init_weights()