import torch
from torch import Tensor

import numpy as np

from scipy.spatial.transform import Rotation as R

class EulerEncoder():
    def __init__(self, device: str = "cpu"):
        pass

    def encode(self, ori: np.ndarray) -> dict[str, np.ndarray]:
        """
        Encode the quaternion to Euler angle
        Args:
            ori (np.ndarray): the quaternion to encode
        
        Returns:
            dict[str, np.ndarray]: the encoded Euler angle
        """
        rotation = R.from_quat(ori, scalar_first=True)
        euler = rotation.as_euler("YXZ", degrees=False)

        return {
            "euler": euler
        }


class EulerDecoder():
    def __init__(self, device: str = "cuda"):
        self.device = device

    def decode_batch(self, ori_pre_dict: dict[str, Tensor]) -> Tensor:
        """
        Decode the Euler angle to quaternion
        Args:
            ori_pre_dict (dict[str, Tensor]): the Euler angle to decode
        
        Returns:
            Tensor: the decoded quaternion
        """
        euler = ori_pre_dict["euler"]
        yaw, pitch, roll = euler.unbind(1)
        half_yaw = yaw * 0.5
        half_pitch = pitch * 0.5
        half_roll = roll * 0.5

        cy = torch.cos(half_yaw)
        sy = torch.sin(half_yaw)
        cp = torch.cos(half_pitch)
        sp = torch.sin(half_pitch)
        cr = torch.cos(half_roll)
        sr = torch.sin(half_roll)
        
        q0 = cy * cp * cr + sy * sp * sr
        q1 = cy * sp * cr + sy * cp * sr
        q2 = sy * cp * cr - cy * sp * sr
        q3 = -sy * sp * cr + cy * cp * sr
        
        return torch.stack([q0, q1, q2, q3], dim=1)