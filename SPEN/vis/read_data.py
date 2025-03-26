from typing import Tuple

import cv2 as cv
import numpy as np
import random

import json


def read_speed_data(image_name: str, config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read the image and its corresponding position, orientation and bounding box
    from the speed dataset.

    Args:
        image_name (str): The name of the image.
        config: The configuration of the dataset.
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The image, position, orientation and bounding box.
    """
    label_path = config.dataset_folder / "train_label.json"
    label = json.load(open(label_path, "r"))
    if not image_name:
        image_name = random.choice(list(label.keys()))
    image_path = config.dataset_folder / "images" / "train" / image_name
    image = cv.imread(str(image_path))
    pos = np.array(label[image_name]["pos"], dtype=np.float32)
    ori = np.array(label[image_name]["ori"], dtype=np.float32)
    box = np.array(label[image_name]["bbox"], dtype=np.int32)
    return image, pos, ori, box


def read_spark_data(image_name: str, config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read the image and its corresponding position and orientation
    from the SPARK dataset.

    Args:
        image_name (str): The name of the image.
        config: The configuration of the dataset.
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The image, position and orientation.
    """
    train_label_path = config.dataset_folder / "train" / "train.json"
    val_label_path = config.dataset_folder / "val" / "val.json"
    train_label = json.load(open(train_label_path, "r"))
    val_label = json.load(open(val_label_path, "r"))
    
    if not image_name:
        image_name = random.choice(list(train_label.keys()))

    if image_name in train_label:
        label = train_label
        image_path = config.dataset_folder / "train" / "images" / label[image_name]["sequence"] / image_name
    else:
        label = val_label
        image_path = config.dataset_folder / "val" / "images" / label[image_name]["sequence"] / image_name
    
    image = cv.imread(str(image_path))
    pos = np.array(label[image_name]["pos"], dtype=np.float32)
    ori = np.array(label[image_name]["ori"], dtype=np.float32)
    box = np.array(label[image_name]["bbox"], dtype=np.int32)

    return image, pos, ori, box