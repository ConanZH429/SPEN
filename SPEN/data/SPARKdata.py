import json
import torch

import cv2 as cv
import numpy as np
import albumentations as A

from threading import Thread
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torchvision.transforms import v2, InterpolationMode
from ..cfg import SPARKConfig
from .augmentation import DropBlockSafe, CropAndPadSafe, CropAndPaste, AlbumentationAug, ZAxisRotation, PerspectiveAug
from .utils import MultiEpochsDataLoader
from ..pose import get_pos_encoder, get_ori_encoder
from ..utils import SPARKCamera

from typing import List, Dict, Tuple


class ImageReader(Thread):
    def __init__(self, image_name: List, image_size: Tuple[int, int], dataset_folder: Path):
        Thread.__init__(self)
        self.image_size = image_size
        self.dataset_folder = dataset_folder
        self.image_name =  image_name
        self.image_dict: dict = {}
    
    def run(self):
        for image_name in tqdm(self.image_name):
            seq = image_name.split(".")[0].split("_")[1]
            image = cv.imread(str(self.dataset_folder / "images" / seq / image_name), cv.IMREAD_GRAYSCALE)
            if self.image_size is not None:
                image = cv.resize(image, (self.image_size[1], self.image_size[0]), interpolation=cv.INTER_LINEAR)
            self.image_dict[image_name] = image
    
    def get_result(self) -> dict:
        return self.image_dict


class SPARKDataset(Dataset):
    """
    The base class for SPARK dataset.
    """
    def __init__(self, config: SPARKConfig, mode: str = "train"):
        super().__init__()
        self.mode = mode
        self.dataset_folder = config.dataset_folder / mode
        self.cache = config.cache
        self.image_size = config.image_size
        self.resize_first = config.resize_first
        self.image_first_size = config.image_first_size
        self.Camera = SPARKCamera(config.image_first_size) if config.resize_first else SPARKCamera(config.image_size)
        # transform the image to tensor
        self.image2tensor = v2.Compose([
            v2.ToImage(),
            v2.Resize(self.image_size, interpolation=InterpolationMode.BILINEAR),
            v2.ToDtype(torch.float32, scale=True)
        ])
        with open(self.dataset_folder / f"{mode}.json", "r") as f:
            self.label = json.load(f)
        self.image_list = list(self.label.keys())
        # transform the value of label to numpy array
        for k in self.label.keys():
            self.label[k]["pos"] = np.array(self.label[k]["pos"], dtype=np.float32)
            self.label[k]["ori"] = np.array(self.label[k]["ori"], dtype=np.float32)
            self.label[k]["bbox"] = np.array(self.label[k]["bbox"], dtype=np.int32)
            if self.resize_first:
                self.label[k]["bbox"] = self.label[k]["bbox"] * self.image_first_size[0] / 1080
                self.label[k]["bbox"] = self.label[k]["bbox"].astype(np.int32)
        # cache the image data
        if self.cache:
            self.image_dict = {}
            # self._cache_image(self.image_list, self.dataset_folder)
            self._cache_image_multithread(self.image_list)
            print(f"Load {mode} images ({self.image_dict[self.image_list[0]].shape}) successfully.")
        self.pos_encoder = get_pos_encoder(config.pos_type, **config.pos_args[config.pos_type])
        self.ori_encoder = get_ori_encoder(config.ori_type, **config.ori_args[config.ori_type])
        if config.pos_type == "DiscreteSpher":
            self.spher_encoder = get_pos_encoder("pher", **config.pos_args["Spher"])
        else:
            self.spher_encoder = None
        if config.ori_type == "DiscreteEuler":
            self.euler_encoder = get_ori_encoder("Euler", **config.ori_args["Euler"])
        else:
            self.euler_encoder = None
        self.len = int(len(self.image_list))
    
    def __len__(self):
        return self.len

    def __getitem__(self, index: int):
        raise NotImplementedError("Subclass of SPARKDataset should implement __getitem__ method.")
    
    def _get_image(self, image_name):
        """
        Get the image data from the the image dict if cache is True,
        otherwise load the image from the disk.

        Args:
            image_name (str): The image name.
        
        Returns:
            np.ndarray: The image data.
        """
        if self.cache:
            return self.image_dict[image_name]
        else:
            seq = self.label[image_name]["sequence"]
            image = cv.imread(str(self.dataset_folder / "images" / seq / image_name), cv.IMREAD_GRAYSCALE)
            if self.resize_first:
                image = cv.resize(image, (self.image_first_size[1], self.image_first_size[0]), interpolation=cv.INTER_LINEAR)
            return image
    
    def _get_label(self, image_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the label data from the label dict.

        Args:
            image_name (str): The image name.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The position, orientation and bounding box.
        """
        label = self.label[image_name]
        return label["pos"], label["ori"], label["bbox"]
    
    def divide_data(self, lst: list, n: int):
        # 将列表lst分为n份，最后不足一份单独一组
        k, m = divmod(len(lst), n)
        return (lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    def _cache_image_multithread(self, image_list: List[str]):
        image_divided = self.divide_data(image_list, 10)
        threads = []
        if self.resize_first:
            for sub_image_name in image_divided:
                threads.append(ImageReader(sub_image_name, self.image_first_size, self.dataset_folder))
        else:
            for sub_image_name in image_divided:
                threads.append(ImageReader(sub_image_name, None, self.dataset_folder))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for thread in threads:
            self.image_dict.update(thread.get_result())


class SPARKTrainDataset(SPARKDataset):
    """
    The dataset class for SPARK train set.
    """
    def __init__(self, config: SPARKConfig):
        super().__init__(config, "train")
        self.crop_and_paste = CropAndPaste(p=config.CropAndPaste_p)
        self.crop_and_pad_safe = CropAndPadSafe(p=config.CropAndPadSafe_p)
        self.drop_block_safe = DropBlockSafe(p=config.DropBlockSafe_p, **config.DropBlockSafe_args)
        self.z_axis_rotation = ZAxisRotation(p=config.ZAxisRotation_p, Camera=self.Camera, **config.ZAxisRotation_args)
        self.persepctive_aug = PerspectiveAug(p=config.Perspective_p,
                                              Camera=self.Camera,
                                              **config.Perspective_args)
        self.albumentation_aug = AlbumentationAug(p=config.AlbumentationAug_p)
    
    def __getitem__(self, index):
        image = self._get_image(self.image_list[index])
        pos, ori, box = self._get_label(self.image_list[index])
        
        # data augmentation
        image = self.crop_and_paste(image, box)
        image = self.crop_and_pad_safe(image, box)
        image = self.drop_block_safe(image, box)
        image, pos, ori, box = self.z_axis_rotation(image, pos, ori, box)
        image, pos, ori, box = self.persepctive_aug(image, pos, ori, box)
        image = self.albumentation_aug(image)

        # transform the image to tensor
        image_tensor = self.image2tensor(image)
        
        label = {
            "image_name": self.image_list[index],
            "pos": pos.astype(np.float32),
            "ori": ori.astype(np.float32),
            "box": box.astype(np.int32),
        }
        # encode the position
        label["pos_encode"] = self.pos_encoder.encode(pos)
        # encode the orientation
        label["ori_encode"] = self.ori_encoder.encode(ori)
        # transform the position to cart
        if self.spher_encoder is not None:
            label["spher"] = self.spher_encoder.encode(pos)
        # transform the orientation to euler
        if self.euler_encoder is not None:
            label["euler"] = self.euler_encoder.encode(ori)

        return image_tensor, image, label


class SPARKValDataset(SPARKDataset):
    """
    The dataset class for SPARK val set.
    """
    def __init__(self, config: SPARKConfig):
        super().__init__(config, "val")
    
    def __getitem__(self, index):
        image = self._get_image(self.image_list[index])
        pos, ori, box = self._get_label(self.image_list[index])

        # transform the image to tensor
        image_tensor = self.image2tensor(image)

        label = {
            "image_name": self.image_list[index],
            "pos": pos.astype(np.float32),
            "ori": ori.astype(np.float32),
        }
        # encode the position
        label["pos_encode"] = self.pos_encoder.encode(pos)
        # encode the orientation
        label["ori_encode"] = self.ori_encoder.encode(ori)

        return image_tensor, image, label


def get_spark_dataloader(config: SPARKConfig):
    data_loader = DataLoader if config.debug else MultiEpochsDataLoader
    data_loader = DataLoader
    train_dataset = SPARKTrainDataset(config)
    val_dataset = SPARKValDataset(config)
    train_loader = data_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        persistent_workers=True,
        pin_memory=True,
        pin_memory_device="cuda",
        prefetch_factor=4
    )
    val_loader = data_loader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        persistent_workers=True,
        pin_memory=True,
        pin_memory_device="cuda",
        prefetch_factor=4
    )
    return train_loader, val_loader