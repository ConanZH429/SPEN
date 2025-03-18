import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


def torch_wasserstein_loss(tensor_a,tensor_b):
    #Compute the first Wasserstein distance between two 1D distributions.
    return(torch_cdf_loss(tensor_a,tensor_b,p=1))

def torch_energy_loss(tensor_a,tensor_b):
    # Compute the energy distance between two 1D distributions.
    return((2**0.5)*torch_cdf_loss(tensor_a,tensor_b,p=2))

def torch_cdf_loss(tensor_a,tensor_b,p=1):
    # last-dimension is weight distribution
    # p is the norm of the distance, p=1 --> First Wasserstein Distance
    # to get a positive weight with our normalized distribution
    # we recommend combining this loss with other difference-based losses like L1

    # normalize distribution, add 1e-14 to divisor to avoid 0/0
    tensor_a = tensor_a / (torch.sum(tensor_a, dim=-1, keepdim=True) + 1e-14)
    tensor_b = tensor_b / (torch.sum(tensor_b, dim=-1, keepdim=True) + 1e-14)
    # make cdf with cumsum
    cdf_tensor_a = torch.cumsum(tensor_a,dim=-1)
    cdf_tensor_b = torch.cumsum(tensor_b,dim=-1)

    # choose different formulas for different norm situations
    if p == 1:
        cdf_distance = torch.sum(torch.abs((cdf_tensor_a-cdf_tensor_b)),dim=-1)
    elif p == 2:
        cdf_distance = torch.sqrt(torch.sum(torch.pow((cdf_tensor_a-cdf_tensor_b),2),dim=-1))
    else:
        cdf_distance = torch.pow(torch.sum(torch.pow(torch.abs(cdf_tensor_a-cdf_tensor_b),p),dim=-1),1/p)

    cdf_loss = cdf_distance.mean()
    return cdf_loss

def torch_validate_distibution(tensor_a,tensor_b):
    # Zero sized dimension is not supported by pytorch, we suppose there is no empty inputs
    # Weights should be non-negetive, and with a positive and finite sum
    # We suppose all conditions will be corrected by network training
    # We only check the match of the size here
    if tensor_a.size() != tensor_b.size():
        raise ValueError("Input weight tensors must be of the same size")


class CrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super(CrossEntropyLoss, self).__init__()
    
    def forward(self, pre: Tensor, label: Tensor):
        return -torch.mean(torch.sum(label * pre, dim=1))


class KLLoss(nn.Module):
    def __init__(self, **kwargs):
        super(KLLoss, self).__init__()
    
    def forward(self, pre, label):
        return F.kl_div(pre, label, reduction="batchmean")


class JSLoss(nn.Module):
    def __init__(self, **kwargs):
        super(JSLoss, self).__init__()
    
    def forward(self, pre, label):
        m = 0.5 * (torch.exp(pre) + label)
        js1 = F.kl_div(pre, m, reduction="batchmean")
        js2 = F.kl_div(torch.log(label+1e-6), m, reduction="batchmean")
        return 0.5 * (js1 + js2)


class WassersteinLoss(nn.Module):
    def __init__(self, **kwargs):
        super(WassersteinLoss, self).__init__()
    
    def forward(self, pre: Tensor, label: Tensor):
        return torch_wasserstein_loss(torch.exp(pre), label)





# CartLoss
class CartLoss(nn.Module):
    loss_dict = {
        "L1": nn.L1Loss,
        "L2": nn.MSELoss,
        "SmoothL1": nn.SmoothL1Loss,
    }

    def __init__(self, loss_type: str = "L2", **kwargs):
        super().__init__()
        if loss_type not in CartLoss.loss_dict:
            raise ValueError(f"Unknown loss type: {loss_type}")
        self.loss = CartLoss.loss_dict[loss_type](**kwargs)
    
    def forward(self, pos_pre_dict: dict[str, Tensor], pos_label_dict: dict[str, Tensor]):
        return {
            "cart_loss": self.loss(pos_pre_dict["cart"], pos_label_dict["cart"])
        }


class SpherLoss(nn.Module):
    loss_dict = {
        "L1": nn.L1Loss,
        "L2": nn.MSELoss,
        "SmoothL1": nn.SmoothL1Loss,
    }

    def __init__(self, loss_type: str = "L2", **kwargs):
        super().__init__()
        if loss_type not in SpherLoss.loss_dict:
            raise ValueError(f"Unknown loss type: {loss_type}")
        self.loss = SpherLoss.loss_dict[loss_type](**kwargs)
    
    def forward(self, pos_pre_dict: dict[str, Tensor], pos_label_dict: dict[str, Tensor]):
        return {
            "spher_loss": self.loss(pos_pre_dict["spher"], pos_label_dict["spher"])
        }


# DiscreteSpherLoss
class DiscreteSpherLoss(nn.Module):
    loss_dict = {
        "L1": nn.L1Loss,
        "CE": CrossEntropyLoss,
        "KL": KLLoss,
        "JS": JSLoss,
        "WassersteinLoss": WassersteinLoss,
    }

    def __init__(self, loss_type: str = "CE", **kwargs):
        super().__init__()
        if loss_type not in DiscreteSpherLoss.loss_dict:
            raise ValueError(f"Unknown loss type: {loss_type}")
        self.loss = DiscreteSpherLoss.loss_dict[loss_type](**kwargs)
    
    def forward(self, ori_pre_dict: dict[str, Tensor], ori_label_dict: dict[str, Tensor]):
        return {
            "r_loss": self.loss(ori_pre_dict["r_encode"], ori_label_dict["r_encode"]),
            "theta_loss": self.loss(ori_pre_dict["theta_encode"], ori_label_dict["theta_encode"]),
            "phi_loss": self.loss(ori_pre_dict["phi_encode"], ori_label_dict["phi_encode"]),
        }


# QuatLoss
class CosSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosSimilarityLoss, self).__init__()
    
    def forward(self, ori_pre: Tensor, ori_label: Tensor):
        ori_inner_dot = torch.abs(torch.sum(ori_pre * ori_label, dim=1))
        ori_inner_dot = torch.clamp(ori_inner_dot, max=1.0, min=-1.0)
        return torch.mean(torch.arccos(ori_inner_dot))


class CosDistanceLoss(nn.Module):
    def __init__(self):
        super(CosDistanceLoss, self).__init__()
    
    def forward(self, ori_pre: Tensor, ori_label: Tensor):
        ori_inner_dot = torch.abs(torch.sum(ori_pre * ori_label, dim=1))
        ori_inner_dot = torch.clamp(ori_inner_dot, max=1.0, min=-1.0)
        return torch.mean(1 - ori_inner_dot)


class ExpCosDistanceLossLoss(nn.Module):
    def __init__(self):
        super(ExpCosDistanceLossLoss, self).__init__()
    
    def forward(self, ori_pre: Tensor, ori_label: Tensor):
        ori_inner_dot = torch.abs(torch.sum(ori_pre * ori_label, dim=1))
        ori_inner_dot = torch.clamp(ori_inner_dot, max=1.0, min=-1.0)
        return torch.mean(torch.exp(1 - ori_inner_dot))


class QuatLoss(nn.Module):
    loss_dict = {
        "Cos": CosSimilarityLoss,
        "CosDistance": CosDistanceLoss,
        "ExpCos": ExpCosDistanceLossLoss,
    }

    def __init__(self, loss_type: str = "Cos", **kwargs):
        super(QuatLoss, self).__init__()
        if loss_type not in QuatLoss.loss_dict:
            raise ValueError(f"Unknown loss type: {loss_type}")
        self.loss = QuatLoss.loss_dict[loss_type](**kwargs)
    
    def forward(self, ori_pre_dict: dict[str, Tensor], ori_label_dict: dict[str, Tensor]):
        return {
            "quat_loss": self.loss(ori_pre_dict["quat"], ori_label_dict["quat"])
        }


# EulerLoss
class EulerLoss(nn.Module):
    loss_dict = {
        "L1": nn.L1Loss,
        "L2": nn.MSELoss,
        "SmoothL1": nn.SmoothL1Loss,
    }

    def __init__(self, loss_type: str = "L1", **kwargs):
        super(EulerLoss, self).__init__()
        if loss_type not in EulerLoss.loss_dict:
            raise ValueError(f"Unknown loss type: {loss_type}")
        self.loss = EulerLoss.loss_dict[loss_type](**kwargs)
    
    def forward(self, ori_pre_dict: dict[str, Tensor], ori_label_dict: dict[str, Tensor]):
        return {
            "euler_loss": self.loss(ori_pre_dict["euler"], ori_label_dict["euler"])
        }



# DiscreteEulerLoss
class DiscreteEulerLoss(nn.Module):
    loss_dict = {
        "L1": nn.L1Loss,
        "CE": CrossEntropyLoss,
        "KL": KLLoss,
        "JS": JSLoss,
        "WassersteinLoss": WassersteinLoss,
    }

    def __init__(self, loss_type: str = "CE", **kwargs):
        super().__init__()
        if loss_type not in DiscreteEulerLoss.loss_dict:
            raise ValueError(f"Unknown loss type: {loss_type}")
        self.loss = DiscreteEulerLoss.loss_dict[loss_type](**kwargs)
    
    def forward(self, ori_pre_dict: dict[str, Tensor], ori_label_dict: dict[str, Tensor]):
        return {
            "yaw_loss": self.loss(ori_pre_dict["yaw_encode"], ori_label_dict["yaw_encode"]),
            "pitch_loss": self.loss(ori_pre_dict["pitch_encode"], ori_label_dict["pitch_encode"]),
            "roll_loss": self.loss(ori_pre_dict["roll_encode"], ori_label_dict["roll_encode"]),
        }


class PosLossFunc(nn.Module):
    def __init__(self, pos_type: str, loss_type: str, **kwargs):
        super().__init__()
        self.pos_type = pos_type
        if pos_type == "Cart":
            self.loss = CartLoss(loss_type, **kwargs)
        elif pos_type == "Spher":
            self.loss = SpherLoss(loss_type, **kwargs)
        elif pos_type == "DiscreteSpher":
            self.loss = DiscreteSpherLoss(loss_type, **kwargs)
        else:
            raise ValueError(f"Unknown pos type: {pos_type}")
    
    def forward(self, pos_pre_dict: dict[str, Tensor], pos_label_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.loss(pos_pre_dict, pos_label_dict)


class OriLossFunc(nn.Module):
    def __init__(self, ori_type: str, loss_type: str, **kwargs):
        super().__init__()
        self.ori_type = ori_type
        if ori_type == "Quat":
            self.loss = QuatLoss(loss_type, **kwargs)
        elif ori_type == "Euler":
            self.loss = EulerLoss(loss_type, **kwargs)
        elif ori_type == "DiscreteEuler":
            self.loss = DiscreteEulerLoss(loss_type, **kwargs)
        else:
            raise ValueError(f"Unknown ori type: {ori_type}")
    
    def forward(self, ori_pre_dict: dict[str, Tensor], ori_label_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.loss(ori_pre_dict, ori_label_dict)