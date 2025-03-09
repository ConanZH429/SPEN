from .blocks import *
from typing import List

# head
class CartHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        in_channels = in_channels // 2
        self.cart_fc = nn.Linear(in_channels, 3, bias=False)
    
    def forward(self, pos_feature: Tensor):
        cart = self.cart_fc(pos_feature)
        return {
            "cart": cart
        }


class SpherHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        in_channels = in_channels // 2
        self.spher_fc = nn.Linear(in_channels, 3, bias=False)
    
    def forward(self, x: Tensor):
        spher = self.spher_fc(x)
        return {
            "spher": spher
        }


class DiscreteSpherHead(nn.Module):
    def __init__(self, in_channels: int, angle_stride: int, r_stride: int, r_max: int, neighbor: int, **kwargs):
        super().__init__()
        in_channels = in_channels // 2
        r_dim = r_max // r_stride + 1 + 2 * neighbor
        theta_dim = 90 // angle_stride + 1 + 2 * neighbor
        phi_dim = 360 // angle_stride + 1 + 2 * neighbor
        self.r_fc = nn.Linear(in_channels, r_dim, bias=False)
        self.theta_fc = nn.Linear(in_channels, theta_dim, bias=False)
        self.phi_fc = nn.Linear(in_channels, phi_dim, bias=False)
    
    def forward(self, pos_feature: Tensor):
        r_encode = self.r_fc(pos_feature)
        theta_encode = self.theta_fc(pos_feature)
        phi_encode = self.phi_fc(pos_feature)
        r_encode = r_encode.type(torch.float32)
        theta_encode = theta_encode.type(torch.float32)
        phi_encode = phi_encode.type(torch.float32)
        r_encode = F.log_softmax(r_encode, dim=-1)
        theta_encode = F.log_softmax(theta_encode, dim=-1)
        phi_encode = F.log_softmax(phi_encode, dim=-1)
        return {
            "r_encode": r_encode,
            "theta_encode": theta_encode,
            "phi_encode": phi_encode
        }


class QuatHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        in_channels = in_channels // 2
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
        in_channels = in_channels // 2
        self.euler_fc = nn.Linear(in_channels, 3, bias=False)
    
    def forward(self, x: Tensor):
        euler = self.euler_fc(x)
        return {
            "euler": euler
        }


class DiscreteEulerHead(nn.Module):
    def __init__(self, in_channels: int, stride: int, neighbor: int, **kwargs):
        super().__init__()
        in_channels = in_channels // 2
        yaw_dim = 360 // stride + 1 + 2 * neighbor
        pitch_dim = 180 // stride + 1 + 2 * neighbor
        roll_dim = 360 // stride + 1 + 2 * neighbor
        self.yaw_fc = nn.Linear(in_channels, yaw_dim, bias=False)
        self.pitch_fc = nn.Linear(in_channels, pitch_dim, bias=False)
        self.roll_fc = nn.Linear(in_channels, roll_dim, bias=False)

    def forward(self, ori_feature: Tensor):
        yaw_encode = self.yaw_fc(ori_feature)
        pitch_encode = self.pitch_fc(ori_feature)
        roll_encode = self.roll_fc(ori_feature)
        yaw_encode = yaw_encode.type(torch.float32)
        pitch_encode = pitch_encode.type(torch.float32)
        roll_encode = roll_encode.type(torch.float32)
        yaw_encode = F.log_softmax(yaw_encode, dim=-1)
        pitch_encode = F.log_softmax(pitch_encode, dim=-1)
        roll_encode = F.log_softmax(roll_encode, dim=-1)
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
    def __init__(self, in_channels: List[int], config):
        super().__init__()
        feature_dim = sum(channels*avg_size**2 for channels, avg_size in zip(in_channels, config.avg_size))
        self.pos_dim = feature_dim
        self.ori_dim = feature_dim
        self.fuse_fc = Mlp(feature_dim, feature_dim // 2, feature_dim // 2, act_layer=MLPAct, bias=False)
        self.avg_size = config.avg_size
        PosHead = Head.pos_head_dict[config.pos_type]
        self.pos_head = PosHead(self.pos_dim, **config.pos_args[config.pos_type])
        OriHead = Head.ori_head_dict[config.ori_type]
        self.ori_head = OriHead(self.ori_dim, **config.ori_args[config.ori_type])
        self.weight_init()

    
    def forward(self, features: List[Tensor]):
        assert len(features) == len(self.avg_size), f"features length {len(features)} != avg_size length {len(self.avg_size)}"
        feature_pooled = [F.adaptive_avg_pool2d(x, avg_size).flatten(1) for x, avg_size in zip(features, self.avg_size)]
        feature_pooled = torch.cat(feature_pooled, dim=1)
        feature_fused = self.fuse_fc(feature_pooled)
        pos_feature = feature_fused
        ori_feature = feature_fused
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