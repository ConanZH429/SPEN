import torch

import torch.nn.functional as F
from torch import linalg as LA

from torch import Tensor
from torchmetrics import Metric
from typing import List


class Loss(Metric):
    """
    Loss metric
    self.loss is the sum of the loss
    self.num_samples is the number of samples
    """
    is_differentiable = True

    def __init__(self):
        super().__init__()
        self.add_state("loss", default=torch.tensor(0.0))
        self.add_state("num_samples", default=torch.tensor(0.0))
    
    def update(self, loss: Tensor, num_samples: int):
        self.loss += loss * num_samples
        self.num_samples += num_samples
    
    def compute(self):
        return self.loss / self.num_samples


class PosLoss(Metric):
    """
    Position loss metric
    """
    is_differentiable = True

    def __init__(self, pos_type: str):
        super().__init__()
        self.add_state("num_samples", default=torch.tensor(0.0))
        if pos_type == "Cart":
            self.add_state("cart_loss", default=torch.tensor(0.0))
            self.update = self.update_cart
            self.compute = self.compute_cart
        elif pos_type == "Spher":
            self.add_state("spher_loss", default=torch.tensor(0.0))
            self.update = self.update_spher
            self.compute = self.compute_spher
        elif pos_type == "DiscreteSpher":
            self.add_state("r_loss", default=torch.tensor(0.0))
            self.add_state("theta_loss", default=torch.tensor(0.0))
            self.add_state("phi_loss", default=torch.tensor(0.0))
            self.update = self.update_discrete_spher
            self.compute = self.compute_discrete_spher
        else:
            raise ValueError(f"Unknown pos type: {pos_type}")
    

    def update(self, pos_loss_dict: dict, num_samples: int):
        raise NotImplementedError
    

    def compute(self):
        raise NotImplementedError

    
    def update_cart(self, pos_loss_dict: dict, num_samples: int):
        self.num_samples += num_samples
        self.cart_loss += pos_loss_dict["cart_loss"] * num_samples
    
    def compute_cart(self):
        return {
            "cart_loss": self.cart_loss / self.num_samples
        }
    

    def update_spher(self, pos_loss_dict: dict, num_samples: int):
        self.num_samples += num_samples
        self.spher_loss += pos_loss_dict["spher_loss"] * num_samples
    
    def compute_spher(self):
        return {
            "spher_loss": self.spher_loss / self.num_samples
        }

    
    def update_discrete_spher(self, pos_loss_dict: dict, num_samples: int):
        self.num_samples += num_samples
        self.r_loss += pos_loss_dict["r_loss"] * num_samples
        self.theta_loss += pos_loss_dict["theta_loss"] * num_samples
        self.phi_loss += pos_loss_dict["phi_loss"] * num_samples
    
    def compute_discrete_spher(self):
        return {
            "r_loss": self.r_loss / self.num_samples,
            "theta_loss": self.theta_loss / self.num_samples,
            "phi_loss": self.phi_loss / self.num_samples
        }


class OriLoss(Metric):
    """
    Orientation loss metric
    """
    is_differentiable = True

    def __init__(self, ori_type: str):
        super().__init__()
        self.add_state("num_samples", default=torch.tensor(0.0))
        if ori_type == "Quat":
            self.add_state("quat_loss", default=torch.tensor(0.0))
            self.update = self.update_quat
            self.compute = self.compute_quat
        elif ori_type == "Euler":
            self.add_state("euler_loss", default=torch.tensor(0.0))
            self.update = self.update_euler
            self.compute = self.compute_euler
        elif ori_type == "DiscreteEuler":
            self.add_state("yaw_loss", default=torch.tensor(0.0))
            self.add_state("pitch_loss", default=torch.tensor(0.0))
            self.add_state("roll_loss", default=torch.tensor(0.0))
            self.update = self.update_discrete_euler
            self.compute = self.compute_discrete_euler
        else:
            raise ValueError(f"Unknown ori type: {ori_type}")
    

    def update(self, ori_loss_dict: dict, num_samples: int):
        raise NotImplementedError


    def compute(self):
        raise NotImplementedError

    
    def update_quat(self, ori_loss_dict: dict, num_samples: int):
        self.num_samples += num_samples
        self.quat_loss += ori_loss_dict["quat_loss"] * num_samples
    
    def compute_quat(self):
        return {
            "quat_loss": self.quat_loss / self.num_samples
        }


    def update_euler(self, ori_loss_dict: dict, num_samples: int):
        self.num_samples += num_samples
        self.euler_loss += ori_loss_dict["euler_loss"] * num_samples
    
    def compute_euler(self):
        return {
            "euler_loss": self.euler_loss / self.num_samples
        }    


    def update_discrete_euler(self, ori_loss_dict: dict, num_samples: int):
        self.num_samples += num_samples
        self.yaw_loss += ori_loss_dict["yaw_loss"] * num_samples
        self.pitch_loss += ori_loss_dict["pitch_loss"] * num_samples
        self.roll_loss += ori_loss_dict["roll_loss"] * num_samples
    
    def compute_discrete_euler(self):
        return {
            "yaw_loss": self.yaw_loss / self.num_samples,
            "pitch_loss": self.pitch_loss / self.num_samples,
            "roll_loss": self.roll_loss / self.num_samples
        }


class PosError(Metric):
    """
    Position error metric
    self.pos_error is the sum of the position error
    self.num_samples is the number of samples
    """
    is_differentiable =True

    def __init__(self):
        super().__init__()
        self.add_state("pos_error", default=torch.tensor(0.0))
        self.add_state("Et", default=torch.tensor(0.0))
        self.add_state("num_samples", default=torch.tensor(0.0))
    
    def update(self, pos_pre: Tensor, pos_label: Tensor, num_samples: int):
        Et = LA.vector_norm(pos_pre - pos_label, dim=1)
        Et_norm = Et / LA.vector_norm(pos_label, dim=1)
        self.Et += torch.sum(Et[Et_norm>=0.002173])
        self.pos_error += torch.sum( Et_norm[Et_norm>=0.002173] )
        self.num_samples += num_samples
        if torch.isnan(self.pos_error):
            print(pos_pre, pos_label)
    
    def compute(self):
        return self.pos_error / self.num_samples, self.Et / self.num_samples



class OriError(Metric):
    """
    Orientation error metric
    self.ori_error is the sum of the orientation error
    self.num_samples is the number of samples
    """
    is_differentiable = True

    def __init__(self):
        super().__init__()
        self.add_state("ori_error", default=torch.tensor(0.0))
        self.add_state("num_samples", default=torch.tensor(0.0))
    
    def update(self, ori_pre: Tensor, ori_label: Tensor, num_samples: int):
        ori_pre_norm = F.normalize(ori_pre, p=2, dim=1)
        ori_label_norm = F.normalize(ori_label, p=2, dim=1)
        ori_inner_dot = torch.abs(torch.sum(ori_pre_norm * ori_label_norm, dim=1))
        ori_inner_dot = torch.clamp(ori_inner_dot, max=1.0, min=-1.0)
        ori_error = torch.rad2deg(2 * torch.arccos(ori_inner_dot))
        self.ori_error += torch.sum(ori_error[ori_error > 0.169])
        self.num_samples += num_samples
        if torch.isnan(self.ori_error):
            print(ori_inner_dot)
    
    def compute(self):
        return self.ori_error / self.num_samples



class Score(Metric):
    is_differentiable = False

    def __init__(self, ALPHA: List[float]):
        super().__init__()
        self.add_state("score", default=torch.tensor(0.0))
        self.ALPHA = ALPHA
    
    def update(self, pos_error: Tensor, ori_error: Tensor):
        self.score = self.ALPHA[0] * pos_error + self.ALPHA[1] * ori_error
    
    def compute(self):
        return self.score