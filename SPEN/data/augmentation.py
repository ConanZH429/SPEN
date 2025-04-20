import cv2 as cv
import numpy as np
import albumentations as A
import random
import math

from scipy.spatial.transform import Rotation as R
from typing import Union, Tuple, List, Optional
from pathlib import Path
from ..utils import SPEEDCamera
from .sunflare import sun_flare
# from albumentations.augmentations.geometric.functional import perspective_bboxes

def warp_box(box, M, width, height):
    # warp points
    xy = np.ones((4, 3))

    xy[:, :2] = box[[0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        4, 2
    )  # 所有点，矩阵变换，求变换后的点
    xy = xy @ M.T  # transform

    xy = (xy[:, :2] / xy[:, 2:3]).reshape(8)
    # create new boxes
    x = xy[[0, 2, 4, 6]]
    y = xy[[1, 3, 5, 7]]
    
    xy = np.array([x.min(), y.min(), x.max(), y.max()])
    # clip boxes
    xy[[0, 2]] = xy[[0, 2]].clip(0, width-1)
    xy[[1, 3]] = xy[[1, 3]].clip(0, height-1)
    return xy.astype(np.float32)


class PerspectiveAug():
    """
    Apply perspective augmentation to an image and
    its corresponding position, orientation and bounding box.
    """
    def __init__(self,
                 max_angle: float, rotation_p: float,
                 max_translation: float, translation_p: float,
                 max_scale: float, scale_p: float,
                 Camera,
                 max_t: int = 5,
                 p: float = 1.0):
        """
        Initialize the Perspective class.
        Args:
            max_angle (float): The maximum rotation angle.
            rotation_p (float): The probability of rotation.
            max_translation (float): The maximum translation percentage along the x-axis and y-axis.
            translation_p (float): The probability of translation.
            max_scale (float): The maximum scale ratio.
            scale_p (float): The probability of scale.
            max_t (int): The maximum number of trials.
            p (float): The probability of applying perspective augmentation.
        """
        self.max_angle = max_angle
        self.rotation_p = rotation_p
        self.max_translation = max_translation
        self.translation_p = translation_p
        self.max_scale = max_scale
        self.scale_p = scale_p
        self.Camera = Camera
        self.max_t = max_t
        self.p = p
        self._h_vector = np.array([0, 0, 1])
    
    def __call__(self, image: np.ndarray, pos: np.ndarray, ori: np.ndarray, box: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply perspective augmentation to an image and its corresponding position, orientation and bounding box.

        Args:
            image (np.ndarray): The image.
            pos (np.ndarray): The position.
            ori (np.ndarray): The orientation.
            box (np.ndarray): The bounding box.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The augmented image, position, orientation and bounding box.
        """
        if np.random.rand() > self.p:
            return image, pos, ori, box

        h, w = image.shape[:2]
        original_area = (box[2] - box[0]) * (box[3] - box[1])

        # Generate initial augmentation matrix
        rotation_matrix = np.eye(3, dtype=np.float32)
        translation_matrix = np.eye(3, dtype=np.float32)
        scale_matrix = np.eye(3, dtype=np.float32)

        # Generate augmentation probability
        rotation_p = np.random.rand()
        translation_p = np.random.rand()
        scale_p = np.random.rand()
        augmentation_p = np.random.rand()

        # Don't to be crazy
        t = 0
        l_area = (1-self.max_scale-0.1) * original_area
        r_area = (1+self.max_scale+0.1) * original_area
        warp_matrix: np.ndarray = np.eye(3, dtype=np.float32)
        while True:
            # Rotation
            if 0.0 <= augmentation_p < 0.333333:
            # if rotation_p < self.rotation_p:
                center = (np.random.uniform(box[0], box[2]), np.random.uniform(box[1], box[3]))
                angle = np.random.uniform(-self.max_angle, self.max_angle)
                rotation_matrix = cv.getRotationMatrix2D(center=center,
                                                         angle=angle,
                                                         scale=1.0)
                rotation_matrix = np.vstack([rotation_matrix, self._h_vector])

            # Scale
            elif 0.333333 <= augmentation_p < 0.666667:
            # if scale_p < self.scale_p:
                s = np.random.uniform(-self.max_scale, self.max_scale)
                scale_matrix = np.array([[1+s, 0, 0],
                                         [0, 1+s, 0],
                                         [0, 0, 1]], dtype=np.float32)

            # Translation
            else:
            # if translation_p < self.translation_p:
                tx = np.random.uniform(-self.max_translation, self.max_translation)
                ty = np.random.uniform(-self.max_translation, self.max_translation)
                translation_matrix = np.array([[1, 0, tx * w],
                                               [0, 1, ty * h],
                                               [0, 0, 1]], dtype=np.float32)
            
            warp_matrix = translation_matrix @ scale_matrix @ rotation_matrix

            box_warpped = warp_box(box, warp_matrix, w, h)
            box_warpped_area = (box_warpped[2] - box_warpped[0]) * (box_warpped[3] - box_warpped[1])

            if l_area <= box_warpped_area <= r_area:
                break
            else:
                t += 1
                if t > self.max_t:
                    return image, pos, ori, box
        
        rotation_matrix = self.Camera.K_image_inv @ warp_matrix @ self.Camera.K_image
        rotation = R.from_matrix(rotation_matrix)
        image_warpped = cv.warpPerspective(image, warp_matrix, (w, h), flags=cv.INTER_LINEAR)

        pos_warpped = rotation_matrix @ pos
        ori_warpped = rotation * R.from_quat(ori, scalar_first=True)
        ori_warpped = ori_warpped.as_quat(canonical=True, scalar_first=True)

        return image_warpped, pos_warpped.astype(np.float32), ori_warpped.astype(np.float32), box_warpped.astype(np.float32)


class ZAxisRotation():
    """
    Rotate the image around the z-axis.
    """
    def __init__(self, max_angle: float, Camera, max_t: int = 5, p: float = 1.0):
        """
        Initialize the ZAxisRotation class.

        Args:
            max_angle (float): The maximum rotation angle. Unit: degree.
            max_t (int): The maximum number of trials.
            Camera: The camera.
            p (float): The probability of applying rotation.
        """
        self.max_angle = max_angle
        self.max_t = max_t
        self.Camera = Camera
        self.p = p
    
    def __call__(self, image: np.ndarray, pos: np.ndarray, ori: np.ndarray, box: np.ndarray, key_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply rotation around the z-axis to an image and its corresponding position, orientation, bounding box and key points.
        Args:
            image (np.ndarray): The image.
            pos (np.ndarray): The position. Shape: (3,).
            ori (np.ndarray): The orientation. Shape: (4,).
            box (np.ndarray): The bounding box. Shape: (4,).
            key_points (np.ndarray): The key points. Shape: (11, 3).
        """
        if random.random() > self.p:
            return image, pos, ori, box, key_points

        h, w = image.shape[:2]
        original_area = (box[2] - box[0]) * (box[3] - box[1])

        t = 0
        r_area = 0.8 * original_area
        while True:
            angle = random.uniform(-self.max_angle, self.max_angle)

            rotation = R.from_euler("YXZ", [0, 0, angle], degrees=True)
            rotation_matrix = rotation.as_matrix()

            warp_matrix = self.Camera.K_image @ rotation_matrix @ self.Camera.K_image_inv

            points_warpped = warp_matrix @ key_points.T     # 3x11
            box_warpped = np.array([points_warpped[0].min(), points_warpped[1].min(),
                                    points_warpped[0].max(), points_warpped[1].max()])
            box_warpped_area = (box_warpped[2] - box_warpped[0]) * (box_warpped[3] - box_warpped[1])
    
            if box_warpped_area >= r_area:
                break
            else:
                t += 1
                if t > self.max_t:
                    return image, pos, ori, box, key_points

        image_warpped = cv.warpPerspective(image, warp_matrix, (w, h), flags=cv.INTER_LINEAR)

        pos_warpped = rotation_matrix @ pos
        ori_warpped = rotation * R.from_quat(ori, scalar_first=True)
        ori_warpped = ori_warpped.as_quat(canonical=True, scalar_first=True)

        return image_warpped, pos_warpped.astype(np.float32), ori_warpped.astype(np.float32), box_warpped.astype(np.int32), points_warpped.astype(np.int32).T

class AlbumentationAug():
    """
    Data augmentation using albumentations.
    """
    def __init__(self, p: float, sunflare_p: float = 0.5):
        """
        Initialize the AlbumentationAug class.

        Args:
            p (float): The probability of applying data augmentation.
        """
        self.p = p
        self.sunflare_p = sunflare_p
        self.aug = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2),
                contrast_limit=(-0.2, 0.2),
                p=p
            ),
            A.OneOf([
                A.MotionBlur(blur_limit=(3, 9)),
                A.MedianBlur(blur_limit=(3, 7)),
                A.GaussianBlur()
            ], p=p),
            A.GaussNoise(
                std_range=(0.05, 0.2),
                mean_range=(0.0, 0.1),
                p=p
            )
        ], p=1)
    
    def __call__(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to the image.

        Args:
            image (np.ndarray): The image.

        Returns:
            np.ndarray: The augmented image.
        """
        image = self.aug(image=image)["image"]
        if random.random() < self.sunflare_p:
            image = sun_flare(
                image=image,
                flare_center=(
                    random.randint(box[0], box[2]),
                    random.randint(box[1], box[3]),
                ),
                src_radius=random.randint(350, 700),
                src_color=random.randint(230, 255),
                angle_range=(0, 1),
                num_circles=random.randint(5, 10),
            )
        return image


class DropBlockSafe():
    """
    Drop blocks on the image.
    """
    def __init__(self, drop_num: int, p: float):
        """
        Initialize the DropBlockSafe class.

        Args:
            drop_num (int): The maximum number of drop blocks.
            p (float): The probability of applying drop blocks
        """
        self.drop_num = drop_num
        self.p = p
    
    def __call__(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        """
        Drop blocks on the image.

        Args:
            image (np.ndarray): The image.
            box (np.ndarray): The bounding box.

        Returns:
            np.ndarray: The image with drop blocks.
        """
        if np.random.rand() > self.p:
            return image
        
        # Get the image size
        h, w = image.shape[:2]
        # Get the bounding box
        x_min, y_min, x_max, y_max = box
        # The number of drop blocks
        drop_num = np.random.randint(1, self.drop_num + 1)

        back_ground = np.random.randint(0, 256, (h, w), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)

        # Drop blocks
        for _ in range(drop_num):
            drop_x_min = np.random.randint(0, w-1)
            drop_y_min = np.random.randint(0, h-1)
            drop_x_max = np.random.randint(drop_x_min, w)
            drop_y_max = np.random.randint(drop_y_min, h)
            mask[drop_y_min:drop_y_max, drop_x_min:drop_x_max] = 1
        
        # Drop the blocks around the bounding box
        mask[y_min:y_max, x_min:x_max] = 0
        image = image * (1 - mask) + back_ground * mask

        return image


class CropAndPadSafe():
    """
    Crop and pad the image to original size.
    """
    def __init__(self, p: float):
        self.p = p
    
    def __call__(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        """
        Crop and pad the image to original size.

        Args:
            image (np.ndarray): The image.
            box (np.ndarray): The bounding box.
        
        Returns:
            np.ndarray: The cropped and padded image.
        """
        if np.random.rand() > self.p:
            return image
        
        h, w = image.shape[:2]
        x_min, y_min, x_max, y_max = box

        # Random crop around the bounding box
        top = np.random.randint(0, y_min) if y_min > 0 else 0
        bottom = np.random.randint(y_max, h) if y_max < h else h
        left = np.random.randint(0, x_min) if x_min > 0 else 0
        right = np.random.randint(x_max, w) if x_max < w else w

        # Crop the image
        cropped_image = image[top:bottom, left:right]
        # Pad the image
        padded_image = cv.copyMakeBorder(cropped_image, top, h-bottom, left, w-right, cv.BORDER_REPLICATE)

        return padded_image


class CropAndPaste():
    """
    Crop and paste the image.
    """
    def __init__(self, p: float):
        """
        Initialize the CropAndPaste class.

        Args:
            p (float): The probability of applying crop and paste.
        """
        self.p = p
        # cache background images
        background_folder = Path(__file__).parent.resolve() / "background"
        self.background_images = []
        self.background_size = []
        for file in background_folder.iterdir():
            self.background_images.append(cv.imread(str(file), cv.IMREAD_GRAYSCALE))
            self.background_size.append(self.background_images[-1].shape[:2])
        self.background_num = len(self.background_images)
    
    def __call__(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        """
        Crop and paste the image.

        Args:
            image (np.ndarray): The image.
            box (np.ndarray): The bounding box.
        
        Returns:
            np.ndarray: The cropped and pasted image.
        """
        if np.random.rand() > self.p:
            return image
        
        h, w = image.shape[:2]
        x_min, y_min, x_max, y_max = box
        target_box = image[y_min:y_max, x_min:x_max]

        # Random choose a background image
        idx = np.random.randint(0, self.background_num)
        background = self.background_images[idx]

        # Random crop the background image
        # Minimum width and height is 1/2 of the image
        # Maximum width and height is shape of the background image
        min_h, min_w = h // 2, w // 2
        max_h, max_w = self.background_size[idx]
        crop_h = np.random.randint(min_h, max_h + 1)
        crop_w = np.random.randint(min_w, max_w + 1)
        h_start = np.random.randint(0, max_h - crop_h + 1)
        w_start = np.random.randint(0, max_w - crop_w + 1)
        background = background[h_start:h_start + crop_h, w_start:w_start + crop_w]

        # Resize the background crop to the shape of the image
        background = cv.resize(background, (w, h))

        # Paste the target box to the background
        background[y_min:y_max, x_min:x_max] = target_box

        return background