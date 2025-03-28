import torch
from torch import Tensor
import torch.nn.functional as F

import numpy as np

from scipy.spatial.transform import Rotation as R

class DiscreteEuler():
    """
    discrete Euler angle probability distribution
    """

    def __init__(self, stride: int, device: str = "cuda"):
        """
        Args:
            stride (int): the stride of the grid
            device (str): the device to use
        """
        self.stride = stride
        self.device = device
        self.yaw_len = int(360 // stride + 1)
        self.pitch_len = int(180 // stride + 1)
        self.roll_len = int(360 // stride + 1)
        self.yaw_range = torch.linspace(0, 360, self.yaw_len) - 180        # -180 ~ 180
        self.pitch_range = torch.linspace(0, 180, self.pitch_len) - 90        # -90 ~ 90
        self.roll_range = torch.linspace(0, 360, self.roll_len) - 180        # -180 ~ 180
        self.yaw_range.requires_grad_(False)
        self.pitch_range.requires_grad_(False)
        self.roll_range.requires_grad_(False)
        self.yaw_range = self.yaw_range.to(device)
        self.pitch_range = self.pitch_range.to(device)
        self.roll_range = self.roll_range.to(device)
        self.yaw_index_dict = {int(yaw // stride): i for i, yaw in enumerate(self.yaw_range)}
        self.pitch_index_dict = {int(pitch // stride): i for i, pitch in enumerate(self.pitch_range)}
        self.roll_index_dict = {int(roll // stride): i for i, roll in enumerate(self.roll_range)}


class DiscreteEulerEncoder(DiscreteEuler):
    def __init__(self, stride: int, device: str = "cuda"):
        """
        Args:
            stride (int): the stride of the grid
            device (str): the device
        """
        super().__init__(stride, device)
    
    def _encode_ori(self, angle: float, angle_len: int, index_dict: dict):
        angle_encode = np.zeros(angle_len, dtype=np.float32)
        
        mean = angle / self.stride
        l, r = int(np.floor(mean)), int(np.ceil(mean))
        if l == r:
            angle_encode[index_dict[l]] = 1
        else:
            angle_encode[index_dict[l]] = (r - mean) / (r - l)
            angle_encode[index_dict[r]] = (mean - l) / (r - l)
        
        return angle_encode

    def encode(self, ori: np.ndarray) -> dict[str, np.ndarray]:
        """
        Encode the quaternion angle to the probability distribution
        Args:
            ori (np.ndarray): the Euler angle in quaternion format
        
        Returns:
            encode (dict[str, np.ndarray]): the probability distribution of yaw, pitch, and roll
        """
        rotation = R.from_quat(ori, scalar_first=True)
        yaw, pitch, roll = rotation.as_euler("YXZ", degrees=True)
        
        yaw_encode = self._encode_ori(yaw, self.yaw_len, self.yaw_index_dict)
        pitch_encode = self._encode_ori(pitch, self.pitch_len, self.pitch_index_dict)
        roll_encode = self._encode_ori(roll, self.roll_len, self.roll_index_dict)

        return {
            "yaw_encode": yaw_encode,
            "pitch_encode": pitch_encode,
            "roll_encode": roll_encode
        }


class DiscreteEulerDecoder(DiscreteEuler):
    def __init__(self, stride: int, device: str = "cuda"):
        """
        Args:
            stride (int): the stride of the grid
            device (str): the device
        """
        super().__init__(stride, device)
    
    def _decode_ori(self, angle_encode: Tensor, angle_range: Tensor):
        return torch.sum(angle_encode * angle_range, dim=1)
    
    def decode_ori(self, yaw_encode: Tensor, pitch_encode: Tensor, roll_encode: Tensor):
        """
        Decode the probability distribution to the Euler angle

        Args:
            yaw_encode (Tensor): the probability distribution of yaw
            pitch_encode (Tensor): the probability distribution of pitch
            roll_encode (Tensor): the probability distribution of roll
        
        Returns:
            ori (Tensor): the Euler angle in quaternion format
        """
        yaw = self._decode_ori(yaw_encode, self.yaw_range)
        pitch = self._decode_ori(pitch_encode, self.pitch_range)
        roll = self._decode_ori(roll_encode, self.roll_range)
        
        rotation = R.from_euler('YXZ', [yaw, pitch, roll], degrees=True)
        ori = rotation.as_quat()
        ori = [ori[3], ori[0], ori[1], ori[2]]
        
        return torch.tensor(ori)
    
    def decode_batch(self, ori_pre_dict: dict[str, Tensor]) -> Tensor:
        """
        Decode the probability distribution to the Euler angle in batch
        
        Args:
            ori_pre_dict (dict[str, Tensor]): the probability distributions for yaw, pitch, and roll
        
        Returns:
            ori (Tensor): the Euler angle in quaternion format
        """
        
        yaw_encode = F.softmax(ori_pre_dict["yaw_encode"], dim=-1)
        pitch_encode = F.softmax(ori_pre_dict["pitch_encode"], dim=-1)
        roll_encode = F.softmax(ori_pre_dict["roll_encode"], dim=-1)

        # yaw_encode = torch.where(yaw_encode < 1e-6, torch.tensor(0, device=self.device), yaw_encode)
        # pitch_encode = torch.where(pitch_encode < 1e-6, torch.tensor(0, device=self.device), pitch_encode)
        # roll_encode = torch.where(roll_encode < 1e-6, torch.tensor(0, device=self.device), roll_encode)

        yaw_decode = torch.sum(yaw_encode * self.yaw_range, dim=1)
        pitch_decode = torch.sum(pitch_encode * self.pitch_range, dim=1)
        roll_decode = torch.sum(roll_encode * self.roll_range, dim=1)

        half_yaw = torch.deg2rad(yaw_decode * 0.5)
        half_pitch = torch.deg2rad(pitch_decode * 0.5)
        half_roll = torch.deg2rad(roll_decode * 0.5)
        
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


class DiscreteEuler2Euler(DiscreteEuler):
    def __init__(self, stride: int, device: str = "cpu"):
        """
        Args:
            stride (int): the stride of the grid
            device (str): the device
        """
        super().__init__(stride, device)
    
    def __call__(self, ori_pre_dict: dict[str, Tensor]) -> Tensor:
        """
        Convert the probability distribution to the Euler angle
        Args:
            ori_pre_dict (dict[str, Tensor]): the probability distributions for yaw, pitch, and roll
        
        Returns:
            Tensor: the Euler angle in quaternion format
        """
        yaw_encode = F.softmax(ori_pre_dict["yaw_encode"], dim=-1)
        pitch_encode = F.softmax(ori_pre_dict["pitch_encode"], dim=-1)
        roll_encode = F.softmax(ori_pre_dict["roll_encode"], dim=-1)

        yaw_decode = torch.sum(yaw_encode * self.yaw_range, dim=1)
        pitch_decode = torch.sum(pitch_encode * self.pitch_range, dim=1)
        roll_decode = torch.sum(roll_encode * self.roll_range, dim=1)
        
        return {
            "euler": torch.deg2rad(torch.stack([yaw_decode, pitch_decode, roll_decode], dim=1))
        }