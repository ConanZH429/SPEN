import torch
from torch import Tensor
import numpy as np
import math
import torch.nn.functional as F

class DiscreteSpher():
    """
    discrete spherical coordinate probability distribution
    """

    def __init__(self, angle_stride: int, r_stride: int, r_max: int, device: str = "cuda"):
        """
        Args:
            angle_stride (int): the stride of the grid in polar and azimuthal angle
            r_stride (int): the stride of the grid in radius            
            r_max (int): maximum radius value
            device (str): the device to use
        """
        self.angle_stride = angle_stride
        self.r_stride = r_stride
        self.r_max = r_max
        self.device = device
        self.theta_len = int(90 // angle_stride + 1)
        self.phi_len = int(360 // angle_stride + 1)
        self.r_len = int(r_max // r_stride + 1)
        self.theta_range = torch.linspace(0, 90, self.theta_len, device=device)
        self.phi_range = torch.linspace(0, 360, self.phi_len, device=device) - 180
        self.r_range = torch.linspace(0, r_max, self.r_len, device=device)
        self.theta_index_dict = {int(theta // angle_stride): i for i, theta in enumerate(self.theta_range)}
        self.phi_index_dict = {int(phi // angle_stride): i for i, phi in enumerate(self.phi_range)}
        self.r_index_dict = {int(r // r_stride): i for i, r in enumerate(self.r_range)}
    

class DiscreteSpherEncoder(DiscreteSpher):
    def __init__(self, angle_stride: int, r_stride: int, r_max: int, device: str = "cuda"):
        """
        Args:
            angle_stride (int): the stride of the grid in polar and azimuthal angle
            r_stride (int): the stride of the grid in radius
            r_max (int): maximum radius value
            device (str): the device
        """
        super().__init__(angle_stride, r_stride, r_max, device)
    
    def _encode_pos(self, x: float, x_len: int, x_stride: int, index_dict: dict):
        x_encode = np.zeros(x_len, dtype=np.float32)

        mean = x / x_stride
        l, r = int(np.floor(mean)), int(np.ceil(mean))
        if l == r:
            x_encode[index_dict[l]] = 1
        else:
            x_encode[index_dict[l]] = (r - mean) / (r - l)
            x_encode[index_dict[r]] = (mean - l) / (r - l)
        
        return x_encode

    def encode(self, pos: np.ndarray) -> dict[str, np.ndarray]:
        """
        encode the position to discrete spherical coordinate probability distribution

        Args:
            pos (np.ndarray): the position in the format of (x, y, z)

        Returns:
            dict[str, np.ndarray]: the encoded position
        """
        x, y, z = pos
        r = math.sqrt(x**2 + y**2 + z**2)
        theta = math.acos(z / r) * 180 / math.pi
        phi = math.atan2(y, x) * 180 / math.pi
        r_encode = self._encode_pos(r, self.r_len, self.r_stride, self.r_index_dict)
        theta_encode = self._encode_pos(theta, self.theta_len, self.angle_stride, self.theta_index_dict)
        phi_encode = self._encode_pos(phi, self.phi_len, self.angle_stride, self.phi_index_dict)
        return {
            "r_encode": r_encode,
            "theta_encode": theta_encode,
            "phi_encode": phi_encode
        }


class DiscreteSpherDecoder(DiscreteSpher):
    def __init__(self, angle_stride: int, r_stride: int, r_max: int, device: str = "cuda"):
        """
        Args:
            angle_stride (int): the stride of the grid in polar and azimuthal angle
            r_stride (int): the stride of the grid in radius
            r_max (int): maximum radius value
            device (str): the device
        """
        super().__init__(angle_stride, r_stride, r_max, device)


    def decode_batch(self, pos_pre_dict: dict[str, Tensor]) -> Tensor:
        """
        decode the discrete spherical coordinate probability distribution to position

        Args:
            pos_pre_dict (dict[str, Tensor]): the probability distributions for radius, polar angle, and azimuthal angle

        Returns:
            np.ndarray: the decoded position in the format of (x, y, z)
        """
        r_encode = F.softmax(pos_pre_dict["r_encode"], dim=-1)
        theta_encode = F.softmax(pos_pre_dict["theta_encode"], dim=-1)
        phi_encode = F.softmax(pos_pre_dict["phi_encode"], dim=-1)
        
        r_decode = torch.sum(r_encode * self.r_range, dim=1)
        theta_decode = torch.sum(theta_encode * self.theta_range, dim=1)
        phi_decode = torch.sum(phi_encode * self.phi_range, dim=1)
        theta_decode = torch.deg2rad(theta_decode)
        phi_decode = torch.deg2rad(phi_decode)
        
        z = r_decode * torch.cos(theta_decode)
        r_sin_theta = r_decode * torch.sin(theta_decode)
        x = r_sin_theta * torch.cos(phi_decode)
        y = r_sin_theta * torch.sin(phi_decode)

        return torch.stack([x, y, z], dim=1)


class DiscreteSpher2Spher(DiscreteSpher):
    def __init__(self, angle_stride: int, r_stride: int, r_max: int, device: str = "cpu"):
        """
        Args:
            angle_stride (int): the stride of the grid in polar and azimuthal angle
            r_stride (int): the stride of the grid in radius
            r_max (int): maximum radius value
            device (str): the device
        """
        super().__init__(angle_stride, r_stride, r_max, device)
    
    def __call__(self, pos_pre_dict: dict[str, Tensor]) -> Tensor:
        """
        convert the probability distribution to the discrete spherical coordinate

        Args:
            pos_pre_dict (dict[str, Tensor]): the probability distributions for radius, polar angle, and azimuthal angle

        Returns:
            Tensor: the discrete spherical coordinate
        """
        r_encode = F.softmax(pos_pre_dict["r_encode"], dim=-1)
        theta_encode = F.softmax(pos_pre_dict["theta_encode"], dim=-1)
        phi_encode = F.softmax(pos_pre_dict["phi_encode"], dim=-1)
        
        r_decode = torch.sum(r_encode * self.r_range, dim=1)
        theta_decode = torch.deg2rad(torch.sum(theta_encode * self.theta_range, dim=1))
        phi_decode = torch.deg2rad(torch.sum(phi_encode * self.phi_range, dim=1))

        return {
            "spher": torch.stack([r_decode, theta_decode, phi_decode], dim=1)
        }