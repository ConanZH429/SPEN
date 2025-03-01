import torch
import math
from torch import Tensor

import numpy as np

class SpherEncoder():
    def __init__(self, r_max: int, device: str = "cpu"):
        self.bias = np.array([0, 0, 180])       # r, theta, phi
        self.scale = np.array([r_max, 90, 360])  # r, theta, phi

    def encode(self, pos: np.ndarray) -> dict[str, np.ndarray]:
        x, y, z = pos
        r = math.sqrt(x**2 + y**2 + z**2)
        theta = math.acos(z / r) * 180 / math.pi
        phi = math.atan2(y, x) * 180 / math.pi
        spher = np.array([r, theta, phi])
        spher = (spher + self.bias) / self.scale
        return {
            "spher": spher
        }
    

class SpherDecoder():
    def __init__(self, r_max: int, device: str = "cuda"):
        self.device = device
        self.scale = torch.tensor([r_max, 90, 360], device=device)  # r, theta, phi
        self.bias = torch.tensor([0, 0, 180], device=device)        # r, theta, phi
    
    def decode_batch(self, pos_pre_dict: dict[str, Tensor]) -> Tensor:
        """
        Decode the spherical coordinate to Cartesian coordinate
        Args:
            pos_pre_dict (dict[str, Tensor]): the spherical coordinate to decode
        """
        spher = pos_pre_dict["spher"] * self.scale - self.bias
        r, theta, phi = spher.unbind(1)
        theta = torch.deg2rad(theta)
        phi = torch.deg2rad(phi)

        z = r * torch.cos(theta)
        r_sin_theta = r * torch.sin(theta)
        x = r_sin_theta * torch.cos(phi)
        y = r_sin_theta * torch.sin(phi)
        return torch.stack([x, y, z], dim=1)
