from .blocks import *
from typing import Optional, List

# neck
class ConvNeck(nn.Module):
    def __init__(self, in_channels: List[int], align_channels: int = 160):
        super().__init__()
        self.conv_p3 = ConvNormAct(in_channels[-3], align_channels, 1, act_layer=ConvAct)
        self.conv_p4 = ConvNormAct(in_channels[-2], align_channels, 1, act_layer=ConvAct)
        self.conv_p5 = ConvNormAct(in_channels[-1], align_channels, 1, act_layer=ConvAct)
        self.out_channels = [align_channels, align_channels, align_channels]
    
    def forward(self, x: List[Tensor]):
        return self.conv_p3(x[-3]), self.conv_p4(x[-2]), self.conv_p5(x[-1])


class IdentityNeck(nn.Module):
    def __init__(self, in_channels: List[int]):
        super().__init__()
        self.out_channels = in_channels[-3:]
    
    def forward(self, x: List[Tensor]):
        return x[-3:]


class TaileNeck(nn.Module):
    def __init__(self, in_channels: List[int], align_channels: int = 460):
        super().__init__()
        self.conv = ConvNormAct(in_channels[-1], align_channels, 1, act_layer=ConvAct)
        self.out_channels = [align_channels]

    def forward(self, x: List[Tensor]):
        return [self.conv(x[-1])]


class PAFPN(nn.Module):
    def __init__(self, in_channels: List[int], align_channels: int = 160):
        super().__init__()
        self.align_p3 = ConvNormAct(in_channels[-3], align_channels, 1, act_layer=ConvAct)
        self.align_p4 = ConvNormAct(in_channels[-2], align_channels, 1, act_layer=ConvAct)
        self.align_p5 = ConvNormAct(in_channels[-1], align_channels, 1, act_layer=ConvAct)

        # FPN
        self.conv5_up = ConvNormAct(align_channels, align_channels, 3, act_layer=ConvAct)
        self.conv4_up = ConvNormAct(align_channels, align_channels, 3, act_layer=ConvAct)
        self.conv3_up = ConvNormAct(align_channels, align_channels, 3, act_layer=ConvAct)

        # PAN
        self.conv3_down = ConvNormAct(align_channels, align_channels, 3, act_layer=ConvAct)
        self.downsample_p3 = ConvNormAct(align_channels, align_channels, 3, stride=2, apply_act=False, apply_norm=False)
        self.conv4_down = ConvNormAct(align_channels, align_channels, 3, act_layer=ConvAct)
        self.downsample_p4 = ConvNormAct(align_channels, align_channels, 3, stride=2, apply_act=False, apply_norm=False)
        self.conv5_down = ConvNormAct(align_channels, align_channels, 3, act_layer=ConvAct)

        self.out_channels = [align_channels, align_channels, align_channels]

    
    def forward(self, x: List[Tensor]):
        """
        illustration of a minimal FPNPAN unit
            P5_0 ---------> P5_1 ---------> P5_2 -------->
                             |                ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
                             |                ↑
                             ↓                |
            P3_0 ---------> P3_1 ---------> P3_2 -------->
        """
        # align
        p3_0 = self.align_p3(x[-3])
        p4_0 = self.align_p4(x[-2])
        p5_0 = self.align_p5(x[-1])

        # FPN
        p5_1 = self.conv5_up(p5_0)
        p4_1 = self.conv4_up(p4_0 + F.interpolate(p5_1, scale_factor=2, mode="nearest"))
        p3_1 = self.conv3_up(p3_0 + F.interpolate(p4_1, scale_factor=2, mode="nearest"))

        # PAN
        p3_2 = self.conv3_down(p3_1)
        p4_2 = self.conv4_down(p4_1 + self.downsample_p3(p3_2))
        p5_2 = self.conv5_down(p5_1 + self.downsample_p4(p4_2))

        return p3_2, p4_2, p5_2



class BiFPN(nn.Module):
    def __init__(self, in_channels: List[int], align_channels: int = 160):
        super().__init__()
        self.eps = torch.tensor(1e-6, dtype=torch.float32)
        # align conv
        self.align_p3 = ConvNormAct(in_channels[-3], align_channels, kernel_size=1, apply_act=False, bias=True)
        self.align_p4 = ConvNormAct(in_channels[-2], align_channels, kernel_size=1, apply_act=False, bias=True)
        self.align_p5 = ConvNormAct(in_channels[-1], align_channels, kernel_size=1, apply_act=False, bias=True)

        # conv layers
        self.conv4_up = nn.Sequential(
            ConvNormAct(align_channels, align_channels, kernel_size=3, groups=align_channels, apply_act=False, apply_norm=False),
            ConvNormAct(align_channels, align_channels, kernel_size=1, act_layer=nn.SiLU)
        )
        self.conv3_up = nn.Sequential(
            ConvNormAct(align_channels, align_channels, kernel_size=3, groups=align_channels, apply_act=False, apply_norm=False),
            ConvNormAct(align_channels, align_channels, kernel_size=1, act_layer=nn.SiLU)
        )
        self.conv4_down = nn.Sequential(
            ConvNormAct(align_channels, align_channels, kernel_size=3, groups=align_channels, apply_act=False, apply_norm=False),
            ConvNormAct(align_channels, align_channels, kernel_size=1, act_layer=nn.SiLU)
        )
        self.conv5_down = nn.Sequential(
            ConvNormAct(align_channels, align_channels, kernel_size=3, groups=align_channels, apply_act=False, apply_norm=False),
            ConvNormAct(align_channels, align_channels, kernel_size=1, act_layer=nn.SiLU)
        )

        # weight
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p3_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

        self.out_channels = [align_channels, align_channels, align_channels]

    def forward(self, x: List[Tensor]):
        """
        illustration of a minimal bifpn unit
            P5_0 -------------------------> P5_2 -------->
               |                              ↑
               |-------------|                |
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |             |              ↑ ↑
               |-------------+--------------| |
                             |                |
                             |--------------| |
                                            ↓ |
            P3_0 -------------------------> P3_2 -------->
        """
        # align
        p3_0 = self.align_p3(x[-3])
        p4_0 = self.align_p4(x[-2])
        p5_0 = self.align_p5(x[-1])

        # weights for P5_0 and P4_0 to P4_1
        p4_w1 = F.relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1) + self.eps)
        p4_1 = self.conv4_up(
            weight[0] * p4_0 + weight[1] * F.interpolate(p5_0, scale_factor=2, mode="nearest")
        )

        # weights for P4_1 and P3_0 to P3_2
        p3_w2 = F.relu(self.p3_w2)
        weight = p3_w2 / (torch.sum(p3_w2) + self.eps)
        p3_2 = self.conv3_up(
            weight[0] * p3_0 + weight[1] * F.interpolate(p4_1, scale_factor=2, mode="nearest")
        )
        
        # weights for P4_0, P4_1 and P3_0 to P4_2
        p4_w2 = F.relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2) + self.eps)
        p4_2 = self.conv4_down(
            weight[0] * p4_0 + weight[1] * p4_1 + weight[2] * F.max_pool2d(p3_2, kernel_size=2, stride=2)
        )

        # weights for P5_0 and P4_2 to P5_2
        p5_w2 = F.relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2) + self.eps)
        p5_2 = self.conv5_down(
            weight[0] * p5_0 + weight[1] * F.max_pool2d(p4_2, kernel_size=2, stride=2)
        )

        return p3_2, p4_2, p5_2


class DensAttFPN(nn.Module):
    def __init__(self, in_channels: List[int], align_channels: int = 160, att_type: Optional[str] = None):
        super().__init__()
        self.align_p2 = ConvNormAct(in_channels[-4], align_channels, 1, act_layer=ConvAct)
        self.align_p3 = ConvNormAct(in_channels[-3], align_channels, 1, act_layer=ConvAct)
        self.align_p4 = ConvNormAct(in_channels[-2], align_channels, 1, act_layer=ConvAct)
        self.align_p5 = ConvNormAct(in_channels[-1], align_channels, 1, act_layer=ConvAct)
        
        # FPN
        # FPN
        self.fuse4_up = AttFuse(align_channels, att_type)
        self.conv4_up = UniversalInvertedResidual(align_channels, align_channels,
                                                  exp_ratio=6, act_layer=ConvAct, layer_scale_init_value=None)
        self.fuse3_up = AttFuse(align_channels, att_type)
        self.conv3_up = UniversalInvertedResidual(align_channels, align_channels,
                                                  exp_ratio=6, act_layer=ConvAct, layer_scale_init_value=None)

        # PAN
        self.downsample_p3 = ConvNormAct(align_channels, align_channels, 3, stride=2, apply_act=False, apply_norm=False)
        self.conv4_down = UniversalInvertedResidual(align_channels, align_channels,
                                                    exp_ratio=6, act_layer=ConvAct, layer_scale_init_value=None)
        self.downsample_p4 = ConvNormAct(align_channels, align_channels, 3, stride=2, apply_act=False, apply_norm=False)
        self.conv5_down = UniversalInvertedResidual(align_channels, align_channels,
                                                    exp_ratio=6, act_layer=ConvAct, layer_scale_init_value=None)

        self.out_channels = [align_channels, align_channels, align_channels]
    
    def forward(self, x: List[Tensor]):
        """
        illustration of a minimal FPNPAN unit
            P5_0 -------------------------> P5_2 -------->
              |-------------|                 ↑
                            ↓                 |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
                            ↑ |               ↑
                       |----| |               |
              |--------|      ↓               |
            P3_0 ---------> P3_1 -------------|---------->
                            ↑
                       |----|
              |--------|
            P2_0
        """
        # align
        p2_0 = self.align_p2(x[-4])
        p3_0 = self.align_p3(x[-3])
        p4_0 = self.align_p4(x[-2])
        p5_0 = self.align_p5(x[-1])

        # FPN
        p4_1 = self.conv4_up(self.fuse4_up(p3_0, p4_0, p5_0))
        p3_1 = self.conv3_up(self.fuse3_up(p2_0, p3_0, p4_0))

        # PAN
        p4_2 = self.conv4_down(p4_1 + self.downsample_p3(p3_1))
        p5_2 = self.conv5_down(p5_0 + self.downsample_p4(p4_2))

        return p3_1, p4_2, p5_2