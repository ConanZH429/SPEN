import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from pathlib import Path

from .head import *
from .neck import *
from ..cfg import SPEEDConfig




class SPEN(nn.Module):
    neck_dict = {
        "IdentityNeck": IdentityNeck,
        "ConvNeck": ConvNeck,
        "TaileNeck": TaileNeck,
        "PAFPN": PAFPN,
        "BiFPN": BiFPN,
        "DensAttFPN": DensAttFPN,
    }

    def __init__(self, config: SPEEDConfig = SPEEDConfig()):
        super().__init__()
        self.avg_size = config.avg_size
        # backbone
        model_name = config.backbone
        bin_folder = config.backbone_args[config.backbone]["bin_folder"]
        bin_path = Path(f"./SPEN/model/timm_weight/{bin_folder}/pytorch_model.bin")
        self.backbone = timm.create_model(model_name,
                                          pretrained=True,
                                          pretrained_cfg_overlay=dict(file=str(bin_path)),
                                          in_chans=1,
                                          features_only=True)
        backbone_out_channels = self.backbone.feature_info.channels()
        if "mobilenetv3" in model_name:
            self.backbone.blocks.pop(-1)
            self.backbone._stage_out_idx = (6, 5, 3, 2, 1)
            if "large" in model_name:
                backbone_out_channels[-1] = 160
            else:
                backbone_out_channels[-1] = 96
        # neck
        Neck = SPEN.neck_dict[config.neck]
        self.neck = Neck(backbone_out_channels, **config.neck_args[config.neck])
        neck_out_channels = self.neck.out_channels[-len(config.avg_size):]
        # head
        self.head = Head(in_channels=neck_out_channels, config=config)

    def forward(self, x):
        feature_map = self.backbone(x)
        feature_map = self.neck(feature_map)
        feature_map = feature_map[-len(self.avg_size):]
        pos_pre_dict, ori_pre_dict = self.head(feature_map)
        return pos_pre_dict, ori_pre_dict