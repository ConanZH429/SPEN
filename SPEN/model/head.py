from .blocks import *
from typing import List, Dict, Any

# head
class CartHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.cart_fc = nn.Linear(in_channels, 3, bias=False)
    
    def forward(self, pos_feature: Tensor):
        cart = self.cart_fc(pos_feature)
        return {
            "cart": cart
        }


class SpherHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.spher_fc = nn.Linear(in_channels, 3, bias=False)
    
    def forward(self, x: Tensor):
        spher = self.spher_fc(x)
        return {
            "spher": spher
        }


class DiscreteSpherHead(nn.Module):
    def __init__(self, in_channels: int, angle_stride: float, r_stride: float, r_max: int, **kwargs):
        super().__init__()
        r_dim = int(r_max // r_stride + 1)
        theta_dim = int(90 // angle_stride + 1)
        phi_dim = int(360 // angle_stride + 1)
        self.r_fc = nn.Linear(in_channels, r_dim)
        self.theta_fc = nn.Linear(in_channels, theta_dim)
        self.phi_fc = nn.Linear(in_channels, phi_dim)
    
    def forward(self, pos_feature: Tensor):
        if isinstance(pos_feature, tuple):
            r_encode = self.r_fc(pos_feature[0])
            theta_encode = self.theta_fc(pos_feature[1])
            phi_encode = self.phi_fc(pos_feature[2])
        else:
            r_encode = self.r_fc(pos_feature)
            theta_encode = self.theta_fc(pos_feature)
            phi_encode = self.phi_fc(pos_feature)
        return {
            "r_encode": r_encode,
            "theta_encode": theta_encode,
            "phi_encode": phi_encode
        }


class QuatHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.quat_fc = nn.Linear(in_channels, 4, bias=False)
    
    def forward(self, x: Tensor):
        quat = self.quat_fc(x)
        quat = F.normalize(quat, p=2, dim=1)
        return {
            "quat": quat
        }


class EulerHead(nn.Module):
    def __init__(self, in_channels: int, **kwargs):
        super().__init__()
        self.euler_fc = nn.Linear(in_channels, 3, bias=False)
    
    def forward(self, x: Tensor):
        euler = self.euler_fc(x)
        return {
            "euler": euler
        }


class DiscreteEulerHead(nn.Module):
    def __init__(self, in_channels: int, stride: float, **kwargs):
        super().__init__()
        yaw_dim = int(360 // stride + 1)
        pitch_dim = int(180 // stride + 1)
        roll_dim = int(360 // stride + 1)
        self.yaw_fc = nn.Linear(in_channels, yaw_dim)
        self.pitch_fc = nn.Linear(in_channels, pitch_dim)
        self.roll_fc = nn.Linear(in_channels, roll_dim)

    def forward(self, ori_feature: Tensor):
        if isinstance(ori_feature, tuple):
            yaw_encode = self.yaw_fc(ori_feature[0])
            pitch_encode = self.pitch_fc(ori_feature[1])
            roll_encode = self.roll_fc(ori_feature[2])
        else:
            yaw_encode = self.yaw_fc(ori_feature)
            pitch_encode = self.pitch_fc(ori_feature)
            roll_encode = self.roll_fc(ori_feature)
        return {
            "yaw_encode": yaw_encode,
            "pitch_encode": pitch_encode,
            "roll_encode": roll_encode
        }

class Head(nn.Module):
    pos_head_dict = {
        "Cart": CartHead,
        "Spher": SpherHead,
        "DiscreteSpher": DiscreteSpherHead,
    }
    ori_head_dict = {
        "Quat": QuatHead,
        "Euler": EulerHead,
        "DiscreteEuler": DiscreteEulerHead
    }
    def __init__(self, pos_type: str, pos_args: Dict[str, Any], ori_type: str, ori_args: Dict[str, Any], fuse: bool = True):
        super().__init__()
        if fuse:
            self.pos_fuse_fc = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim),
                MLPAct(inplace=True),
            )
            self.ori_fuse_fc = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim),
                MLPAct(inplace=True),
            )
        else:
            self.pos_fuse_fc = nn.Identity()
            self.ori_fuse_fc = nn.Identity()
        PosHead = Head.pos_head_dict[pos_type]
        self.pos_head = PosHead(self.feature_dim, **pos_args[pos_type])
        OriHead = Head.ori_head_dict[ori_type]
        self.ori_head = OriHead(self.feature_dim, **ori_args[ori_type])
        self.weight_init()
    
    def forward(self, features: List[Tensor]):
        assert len(self.fm2vect) == len(features), f"fm2vect length {len(self.fm2vect)} != features length {len(features)}"
        features_vect = [fm2vect(fm) for fm2vect, fm in zip(self.fm2vect, features)]
        pos_feature = torch.cat([vect[0].flatten(1) for vect in features_vect], dim=1)
        ori_feature = torch.cat([vect[1].flatten(1) for vect in features_vect], dim=1)
        pos_feature = self.pos_fuse_fc(pos_feature)
        ori_feature = self.ori_fuse_fc(ori_feature)
        pos_pre_dict = self.pos_head(pos_feature)
        ori_pre_dict = self.ori_head(ori_feature)
        return pos_pre_dict, ori_pre_dict
    
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class AvgPoolHead(Head):
    def __init__(self,
                 in_channels: List[int],
                 head_args: Dict[str, Any],
                 pos_type: str,
                 pos_args: Dict[str, Any],
                 ori_type: str,
                 ori_args: Dict[str, Any]):
        assert len(in_channels) == len(head_args["pool_size"]), f"in_channels length {len(in_channels)} != pool_size length {len(head_args['pool_size'])}"
        pool_size = head_args["pool_size"]
        if isinstance(pool_size[0], int):
            self.feature_dim = sum(channels * ps**2 for channels, ps in zip(in_channels, pool_size))
        elif isinstance(pool_size[0], tuple):
            self.feature_dim = sum(channels * ps[0]*ps[1] for channels, ps in zip(in_channels, pool_size))
        super().__init__(pos_type, pos_args, ori_type, ori_args)
        self.fm2vect = nn.ModuleList([AvgPool(ps) for ps in pool_size])


class MaxPoolHead(Head):
    def __init__(self,
                 in_channels: List[int],
                 head_args: Dict[str, Any],
                 pos_type: str,
                 pos_args: Dict[str, Any],
                 ori_type: str,
                 ori_args: Dict[str, Any]):
        assert len(in_channels) == len(head_args["pool_size"]), f"in_channels length {len(in_channels)} != pool_size length {len(head_args['pool_size'])}"
        pool_size = head_args["pool_size"]
        if isinstance(pool_size[0], int):
            self.feature_dim = sum(channels * ps**2 for channels, ps in zip(in_channels, pool_size))
        elif isinstance(pool_size[0], tuple):
            self.feature_dim = sum(channels * ps[0] * ps[1] for channels, ps in zip(in_channels, pool_size))
        super().__init__(pos_type, pos_args, ori_type, ori_args)
        self.fm2vect = nn.ModuleList([MaxPool(ps) for ps in pool_size])


class MixPoolHead(Head):
    def __init__(self,
                 in_channels: List[int],
                 head_args: Dict[str, Any],
                 pos_type: str,
                 pos_args: Dict[str, Any],
                 ori_type: str,
                 ori_args: Dict[str, Any]):
        assert len(in_channels) == len(head_args["pool_size"]), f"in_channels length {len(in_channels)} != pool_size length {len(head_args['pool_size'])}"
        pool_size = head_args["pool_size"]
        if isinstance(pool_size[0], int):
            self.feature_dim = sum(channels * ps**2 for channels, ps in zip(in_channels, pool_size))
        elif isinstance(pool_size[0], tuple):
            self.feature_dim = sum(channels * ps[0] * ps[1] for channels, ps in zip(in_channels, pool_size))
        super().__init__(pos_type, pos_args, ori_type, ori_args)
        self.fm2vect = nn.ModuleList([MixPool(ps, head_args["weighted_learnable"]) for ps in pool_size])


class SPPHead(Head):
    def __init__(self,
                 in_channels: List[int],
                 head_args: Dict[str, Any],
                 pos_type: str,
                 pos_args: Dict[str, Any],
                 ori_type: str,
                 ori_args: Dict[str, Any]):
        assert len(in_channels) == len(head_args["pool_size"]), f"in_channels length {len(in_channels)} != pool_size length {len(head_args['pool_size'])}"
        pool_size = head_args["pool_size"]
        if isinstance(pool_size[0][0], int):
            pool_s = [sum([ps**2 for ps in pool]) for pool in pool_size]
            self.feature_dim = sum(channels * ps for channels, ps in zip(in_channels, pool_s))
        elif isinstance(pool_size[0][0], tuple):
            pool_s = [sum([ps[0]*ps[1] for ps in pool]) for pool in pool_size]
            self.feature_dim = sum(channels * ps for channels, ps in zip(in_channels, pool_s))
        super().__init__(pos_type, pos_args, ori_type, ori_args)
        self.fm2vect = nn.ModuleList([SPP(ps, head_args["mode"]) for ps in pool_size])


class MHAHead(Head):
    def __init__(self,
                 in_channels: List[int],
                 head_args: Dict[str, Any],
                 pos_type: str,
                 pos_args: Dict[str, Any],
                 ori_type: str,
                 ori_args: Dict[str, Any]):
        assert len(in_channels) == len(head_args["pool_size"]), f"in_channels length {len(in_channels)} != pool_size length {len(head_args['pool_size'])}"
        patch_size = head_args["patch_size"]
        pool_size = head_args["pool_size"]
        if "spp" in head_args["pool_mode"]:
            if isinstance(pool_size[0][0], int):
                pool_s = [sum([ps**2 for ps in pool]) for pool in pool_size]
                self.feature_dim = sum(channels * ps for channels, ps in zip(in_channels, pool_s))
            elif isinstance(pool_size[0][0], tuple):
                pool_s = [sum([ps[0]*ps[1] for ps in pool]) for pool in pool_size]
                self.feature_dim = sum(channels * ps for channels, ps in zip(in_channels, pool_s))
        else:
            if isinstance(pool_size[0], int):
                self.feature_dim = sum(channels * ps**2 for channels, ps in zip(in_channels, pool_size))
            elif isinstance(pool_size[0], tuple):
                self.feature_dim = sum(channels * ps[0]*ps[1] for channels, ps in zip(in_channels, pool_size))
        super().__init__(pos_type, pos_args, ori_type, ori_args)
        self.fm2vect = nn.ModuleList([MHAPool(ic, pas, head_args["embedding_mode"], pos, head_args["pool_mode"], head_args["num_heads"]) for ic, pas, pos in zip(in_channels, patch_size, pool_size)])


class TokenHead(Head):
    def __init__(self,
                 in_channels: List[int],
                 head_args: Dict[str, Any],
                 pos_type: str,
                 pos_args: Dict[str, Any],
                 ori_type: str,
                 ori_args: Dict[str, Any]):
        assert len(in_channels) == len(head_args["patch_size"]), f"in_channels length {len(in_channels)} != patch_size length {len(head_args['patch_size'])}"
        patch_size = head_args["patch_size"]
        self.feature_dim = sum(in_channels)
        super().__init__(pos_type, pos_args, ori_type, ori_args, False)
        self.fm2vect = nn.ModuleList([TokenFeature(ic, ps, head_args["embedding_mode"], head_args["num_heads"], head_args["num_layers"], head_args["learnable_token_num"]) for ic, ps in zip(in_channels, patch_size)])
        if head_args["learnable_token_num"] == 6:
            self.forward = self.forward_advance
    
    def forward_advance(self, features: List[Tensor]):
        assert len(self.fm2vect) == len(features), f"fm2vect length {len(self.fm2vect)} != features length {len(features)}"
        features_vect = [fm2vect(fm) for fm2vect, fm in zip(self.fm2vect, features)]
        r_feature = torch.cat([vect[:, :, 0].flatten(1) for vect in features_vect], dim=1)
        theta_feature = torch.cat([vect[:, :, 1].flatten(1) for vect in features_vect], dim=1)
        phi_feature = torch.cat([vect[:, :, 2].flatten(1) for vect in features_vect], dim=1)
        yaw_feature = torch.cat([vect[:, :, 3].flatten(1) for vect in features_vect], dim=1)
        pitch_feature = torch.cat([vect[:, :, 4].flatten(1) for vect in features_vect], dim=1)
        roll_feature = torch.cat([vect[:, :, 5].flatten(1) for vect in features_vect], dim=1)
        pos_pre_dict = self.pos_head((r_feature, theta_feature, phi_feature))
        ori_pre_dict = self.ori_head((yaw_feature, pitch_feature, roll_feature))
        return pos_pre_dict, ori_pre_dict