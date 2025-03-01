import matplotlib.pyplot as plt
import numpy as np
import json
import cv2 as cv
from scipy.spatial.transform import Rotation as R
from typing import Union

from ..utils import SPEEDCamera, SPARKCamera
from ..cfg import SPEEDConfig, SPARKConfig
from .utils import show_image
from .read_data import read_speed_data, read_spark_data

def display_axis(image: np.ndarray, pos: np.ndarray, ori: np.ndarray, alpha: float = 1.0, show: bool = False, camera: Union[SPEEDCamera, SPARKCamera] = None) -> np.ndarray:
    """
    Display the axis on the image.

    Args:
        pos (np.ndarray): The position of the axis.
        ori (np.ndarray): The orientation of the axis.
        alpha (float): The transparency of the axis. Default: 1.0.
        show (bool): Whether to show the image. Default: False.
    
    Returns:
        np.ndarray: The image with the axis.
    """
    def project(pos: np.ndarray, ori: np.ndarray, Camera) -> np.ndarray:
        points_world = np.array([[0, 0, 0, 1],
                                 [1, 0, 0, 1],
                                 [0, 1, 0, 1],
                                 [0, 0, 1, 1]]).T

        rotation = R.from_quat(ori, scalar_first=True)
        extrinsic_mat = np.hstack((rotation.as_matrix(), pos.reshape(3, 1)))
        points_cam = extrinsic_mat @ points_world
        points_cam = points_cam / points_cam[2]
        points_image = Camera.K @ points_cam
        return points_image[0], points_image[1]
    
    xa, ya = project(pos, ori, camera)
    origin = (int(xa[0]), int(ya[0]))
    x_axis = (int(xa[1]), int(ya[1]))
    y_axis = (int(xa[2]), int(ya[2]))
    z_axis = (int(xa[3]), int(ya[3]))
    if alpha < 1.0:
        axis_mask = np.zeros_like(image, dtype=np.uint8)
        axis_mask = cv.arrowedLine(axis_mask, origin, x_axis, (255, 0, 0), 4, cv.LINE_AA)
        axis_mask = cv.arrowedLine(axis_mask, origin, y_axis, (0, 255, 0), 4, cv.LINE_AA)
        axis_mask = cv.arrowedLine(axis_mask, origin, z_axis, (0, 0, 255), 4, cv.LINE_AA)
        image = cv.addWeighted(image, 1, axis_mask, alpha, 0)
    else:
        image = cv.arrowedLine(image, origin, x_axis, (255, 0, 0), 4, cv.LINE_AA)
        image = cv.arrowedLine(image, origin, y_axis, (0, 255, 0), 4, cv.LINE_AA)
        image = cv.arrowedLine(image, origin, z_axis, (0, 0, 255), 4, cv.LINE_AA)
    if show:
        show_image(image)
    return image


def display_box(image: np.ndarray, box: np.ndarray, show: bool = False) -> np.ndarray:
    """
    Display the bounding box on the image.

    Args:
        box (np.ndarray): The bounding box.
        alpha (float): The transparency of the bounding box. Default: 1.0.
        show (bool): Whether to show the image. Default: False.
    
    Returns:
        np.ndarray: The image with the bounding box.
    """
    x1, y1, x2, y2 = box
    image = cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    if show:
        show_image(image)
    return image


def display_image(image_name: str = None,
                  image: Union[np.ndarray, None] = None,
                  display_label_axis: bool = True,
                  display_pre_axis: bool = False,
                  display_box: bool = False,
                  dataset_type: str = "SPEED",
                  save_path: Union[None, str] = None,
                  pos: Union[np.ndarray, None] = None,
                  ori: Union[np.ndarray, None] = None) -> None:
    """
    Display the image with the label axis and the predicted axis.

    Args:
        image_path (str): The path of the image. Default: None.
        image (Union[np.ndarray, None]): The image to display. Default: None.
        display_label_axis (bool): Whether to display the label axis. Default: True.
        display_pre_axis (bool): Whether to display the predicted axis. Default: False.
        display_box (bool): Whether to display the bounding box. Default: False.
        display_dataset (str): The dataset to display. Default: "SPEED".
        save_path (Union[None, str]): The path to save the image. Default: None.
        pos (Union[np.ndarray, None]): The position of the axis. Default: None.
        ori (Union[np.ndarray, None]): The orientation of the axis. Default: None.
    
    Returns:
        None
    """
    # Read the image and the label
    if dataset_type == "SPEED":
        config = SPEEDConfig()
        image_to_show, pos_label, ori_label, box = read_speed_data(image_name, config=config)
        Camera = SPEEDCamera
    elif dataset_type == "SPARK":
        config = SPARKConfig()
        image_to_show, pos_label, ori_label = read_spark_data(image_name, config=config)
        Camera = SPARKCamera
    else:
        raise ValueError("Invalid dataset!")
    image_to_show = image_to_show if image is None else image
    image_to_show = cv.cvtColor(image_to_show, cv.COLOR_GRAY2BGR)
    pos_label = pos_label if pos is None else pos
    ori_label = ori_label if ori is None else ori

    if display_label_axis:
        image_to_show = display_axis(image_to_show, pos_label, ori_label, camera=Camera)
    if display_pre_axis:
        pass
    if display_box:
        pass
    if save_path:
        cv.imwrite(save_path, image_to_show)
    show_image(image_to_show)