from torch.utils.data import DataLoader
import numpy as np
from scipy.spatial.transform import Rotation as R
from ..utils import SPEEDCamera

def points2box(points: np.ndarray, image_size: tuple) -> np.ndarray:
    """
    Args:
        points: (N, 3)
    """
    box = np.array(
        [
            max(points[:, 0].min(), 0),                 # x_min
            max(points[:, 1].min(), 0),                 # y_min
            min(points[:, 0].max(), image_size[1]),     # x_max
            min(points[:, 1].max(), image_size[0]),     # y_max
        ],
        dtype=np.int32
    )
    return box

def world2image(pos: np.ndarray, ori: np.ndarray, Camera: SPEEDCamera,
                points: np.ndarray) -> np.ndarray:
    points_world = np.hstack((points, np.ones((points.shape[0], 1)))).T
    rotation = R.from_quat(ori, scalar_first=True)
    extrinsic_mat = np.hstack((rotation.as_matrix(), pos.reshape(3, 1)))
    points_cam = extrinsic_mat @ points_world
    points_cam = points_cam / points_cam[2]
    points_image = Camera.K_image @ points_cam  # 3x11
    return points_image.T

class MultiEpochsDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)