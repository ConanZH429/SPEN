import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


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
        "CE": CrossEntropyLoss,
        "KL": KLLoss,
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
        "CE": CrossEntropyLoss,
        "KL": KLLoss,
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


class DynamicWeightAverageLoss(nn.Module):
    def __init__(self, num_tasks, T):
        super(DynamicWeightAverageLoss, self).__init__()
        self.num_tasks = num_tasks
        self.T = T
        self.last_loss_dict = {}
    
    def forward(self, loss_dict: dict[str, Tensor]) -> Tensor:
        if not self.last_loss_dict:
            weight_dict = {
                k: torch.tensor(1.0, device="cuda") for k in loss_dict
            }
        else:
            w_i_dict = {
                k: loss_dict[k] / self.last_loss_dict[k] for k in loss_dict
            }
            exp_sum = torch.tensor(0.0, device="cuda")
            for k in w_i_dict:
                exp_sum += torch.exp(w_i_dict[k] / self.T)
            weight_dict = {
                k: self.num_tasks * torch.exp(w_i_dict[k] / self.T) / exp_sum for k in w_i_dict
            }
        loss = torch.tensor(0.0, device="cuda")
        for k in loss_dict:
            loss += weight_dict[k] * loss_dict[k]
        self.last_loss_dict = {k: loss_dict[k].detach() for k in loss_dict}
        return loss




class GradNormLoss(nn.Module):
    def __init__(self, num_tasks, alpha=1.5):
        """
        GradNormLoss 类用于实现 GradNorm 损失函数。
        
        参数:
        num_tasks (int): 任务的数量。
        alpha (float): 损失项的比例系数，默认为 1.5。
        """
        super(GradNormLoss, self).__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha
        # 初始化每个任务的权重，初始值为 1，并设置为可训练参数
        self.weights = nn.Parameter(torch.ones(num_tasks, requires_grad=True))

    def forward(self, losses, model):
        """
        前向传播函数，计算总的损失。

        参数:
        losses (list): 包含每个任务损失项的列表。
        model (torch.nn.Module): 用于计算梯度的模型。

        返回:
        torch.Tensor: 总的损失值。
        """
        # 计算每个损失项的梯度范数
        grads = []
        for i, loss in enumerate(losses):
            # 计算当前损失项对模型参数的梯度
            grad = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
            # 计算梯度的范数
            grad_norm = torch.norm(torch.stack([torch.norm(g) for g in grad]))
            # 将梯度范数添加到列表中
            grads.append(grad_norm)
        
        # 计算初始梯度范数
        initial_grad_norm = torch.stack(grads).mean()

        # 计算每个损失项的梯度范数相对于初始梯度范数的比例
        relative_grad_norms = [grad / initial_grad_norm for grad in grads]

        # 计算梯度范数的平均值
        avg_relative_grad_norm = sum(relative_grad_norms) / len(relative_grad_norms)

        # 计算梯度范数差异的平方
        grad_norm_diff = [(relative_grad_norm - avg_relative_grad_norm) ** 2 for relative_grad_norm in relative_grad_norms]

        # 计算 GradNorm 损失
        gradnorm_loss = sum(grad_norm_diff) * self.alpha

        # 计算加权后的总损失
        weighted_losses = [w * loss for w, loss in zip(self.weights, losses)]
        total_loss = sum(weighted_losses) + gradnorm_loss

        return total_loss