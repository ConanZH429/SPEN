import torch
from torch import Tensor
import numpy as np

class QuatEncoder:
    def __init__(self):
        pass

    def encode(self, ori: np.ndarray):
        return {
            "quat": ori
        }


class QuatDecoder:
    def __init__(self):
        pass

    def decode_batch(self, ori_pre_dict: dict[str, Tensor]) -> Tensor:
        return ori_pre_dict["quat"]