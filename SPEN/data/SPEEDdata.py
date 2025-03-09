import json
import torch

import lightning as L
import cv2 as cv
import numpy as np
import albumentations as A

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from random import shuffle
from torchvision.transforms import v2
from ..cfg import SPEEDConfig
from .augmentation import DropBlockSafe, CropAndPadSafe, CropAndPaste, AlbumentationAug, ZAxisRotation, PerspectiveAug
from .utils import MultiEpochsDataLoader
from ..pose import get_pos_encoder, get_ori_encoder
from ..utils import SPEEDCamera


from typing import List, Dict, Tuple



def SPEED_split_dataset(config: SPEEDConfig = SPEEDConfig()):
    """
    Split the SPEED dataset into train and val set.

    Args:
        config (SPEEDConfig): The config object.
    
    Returns:
        None
    """
    dataset_folder = config.dataset_folder
    train_txt_path = dataset_folder / "train.txt"
    val_txt_path = dataset_folder / "val.txt"
    train_json_path = dataset_folder / "train_label.json"
    val_json_path = dataset_folder / "val_label.json"
    if train_txt_path.exists() and val_txt_path.exists():
        if train_json_path.exists() and val_json_path.exists():
            return None
    
    with open(dataset_folder / "train.json", "r") as f:
        label = json.load(f)
    for key in label.keys():
        label[key]["d"] = np.linalg.norm(np.array(label[key]["pos"]))
    d_list = [0, 10, 20, 30, 40, 50]
    d_count = [0] * 6
    count_dict_key = ["0-10", "10-20", "20-30", "30-40", "40-50"]
    count_dict = {
        "0-10": [],
        "10-20": [],
        "20-30": [],
        "30-40": [],
        "40-50": []
    }
    for key in label.keys():
        for i in range(5):
            if label[key]["d"] >= d_list[i] and label[key]["d"] < d_list[i+1]:
                d_count[i] += 1
                count_dict[count_dict_key[i]].append(key)
    for key in count_dict.keys():
        shuffle(count_dict[key])
    train_count_all = 10200
    val_count_all = 1800
    train_list = count_dict["40-50"]
    val_list = []
    train_list = []
    val_list = []
    train_count = int(len(count_dict["30-40"]) * 0.85)
    train_list = train_list + count_dict["30-40"][:train_count]
    val_list = val_list + count_dict["30-40"][train_count:]
    train_count = int(len(count_dict["20-30"]) * 0.85)
    train_list = train_list + count_dict["20-30"][:train_count]
    val_list = val_list + count_dict["20-30"][train_count:]
    train_count = int(len(count_dict["10-20"]) * 0.85)
    train_list = train_list + count_dict["10-20"][:train_count]
    val_list = val_list + count_dict["10-20"][train_count:]
    train_count = train_count_all - len(train_list)
    train_list = train_list + count_dict["0-10"][:train_count]
    val_list = val_list + count_dict["0-10"][train_count:]
    train_label = {k: label[k] for k in train_list}
    val_label = {k: label[k] for k in val_list}
    with open(train_txt_path, "w") as f:
        f.write("\n".join(train_list))
    with open(val_txt_path, "w") as f:
        f.write("\n".join(val_list))
    with open(dataset_folder / "train_label.json", "w") as f:
        json.dump(train_label, f)
    with open(dataset_folder / "val_label.json", "w") as f:
        json.dump(val_label, f)
    return None




class SPEEDDataset(Dataset):
    """
    The base class for SPEED dataset.
    """
    def __init__(self, config: SPEEDConfig = SPEEDConfig(), mode: str = "train"):
        super().__init__()
        SPEED_split_dataset(config)
        self.dataset_folder = config.dataset_folder
        self.cache = config.cache
        self.resize_first = config.resize_first
        with open(self.dataset_folder / f"{mode}_label.json", "r") as f:
            self.label = json.load(f)
        self.image_list = list(self.label.keys())
        self.ratio = config.val_ratio if mode == "val" else config.train_ratio
        # transform the value of label to numpy array
        for k in self.label.keys():
            self.label[k]["pos"] = np.array(self.label[k]["pos"], dtype=np.float32)
            self.label[k]["ori"] = np.array(self.label[k]["ori"], dtype=np.float32)
            self.label[k]["bbox"] = np.array(self.label[k]["bbox"], dtype=np.int32)
        # cache the image data
        if self.cache:
            self._cache_image(self.image_list, self.dataset_folder)
        # resize the image
        self.resize = A.Compose([A.Resize(*config.image_size, interpolation=cv.INTER_LINEAR, p=1.0)],
                                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]))
        if config.resize_first:
            self.resize_first = A.Compose([A.Resize(*config.image_first_size, interpolation=cv.INTER_LINEAR, p=1.0)],
                                          bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]))
        # transform the image to tensor
        self.image2tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        self.pos_encoder = get_pos_encoder(config.pos_type, **config.pos_args[config.pos_type])
        self.ori_encoder = get_ori_encoder(config.ori_type, **config.ori_args[config.ori_type])
        self.len = int(len(self.image_list))
    
    def __len__(self) -> int:
        return self.len
    
    def __getitem__(self, index: int):
        raise NotImplementedError("Subclass of SPEEDDataset should implement __getitem__ method.")

    def _get_image(self, image_name: str) -> np.ndarray:
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
            return cv.imread(str(self.dataset_folder / "images" / "train" / image_name), cv.IMREAD_GRAYSCALE)
    
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

    def _cache_image(self, image_list: List[str], dataset_folder: Path) -> Dict[str, np.ndarray]:
        """
        Cache the image data.

        Args:
            image_list (List[str]): The list of image names.
            dataset_folder (Path): The dataset folder.
        
        Returns:
            Dict[str, np.ndarray]: The image dict.
        """
        self.image_dict = {}
        with tqdm(image_list) as tbar:
            for image_name in tbar:
                tbar.set_postfix_str(f"Loading {image_name}")
                self.image_dict[image_name] = cv.imread(str(dataset_folder / "images" / "train" / image_name), cv.IMREAD_GRAYSCALE)
        return self.image_dict
    
    def _resize_image(self, image: np.ndarray, box: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resize the image and the bounding box.

        Args:
            image (np.ndarray): The image data.
            box (np.ndarray): The bounding box.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: The resized image and bounding box.
        """
        transformed = self.resize(image=image, bboxes=box.reshape(1, 4), category_ids=[1])
        return transformed["image"], transformed["bboxes"].reshape(4).astype(np.int32)

    def _resize_image_first(self, image: np.ndarray, box: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resize the image and the bounding box.

        Args:
            image (np.ndarray): The image data.
            box (np.ndarray): The bounding box.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: The resized image and bounding box.
        """
        transformed = self.resize_first(image=image, bboxes=box.reshape(1, 4), category_ids=[1])
        return transformed["image"], transformed["bboxes"].reshape(4).astype(np.int32)
        



class SPEEDTrainDataset(SPEEDDataset):
    """
    The dataset class for SPEED train set.
    """
    def __init__(self, config: SPEEDConfig = SPEEDConfig()):
        super().__init__(config, "train")
        self.crop_and_paste = CropAndPaste(p=config.CropAndPaste_p)
        self.crop_and_pad_safe = CropAndPadSafe(p=config.CropAndPadSafe_p)
        self.drop_block_safe = DropBlockSafe(p=config.DropBlockSafe_p, **config.DropBlockSafe_args)
        self.z_axis_rotation = ZAxisRotation(p=config.ZAxisRotation_p, Camera=SPEEDCamera, **config.ZAxisRotation_args)
        self.persepctive_aug = PerspectiveAug(p=config.Perspective_p,
                                              Camera=SPEEDCamera,
                                              **config.Perspective_args)
        self.albumentation_aug = AlbumentationAug(p=config.AlbumentationAug_p)

    def __getitem__(self, index):
        image = self._get_image(self.image_list[index])
        pos, ori, box = self._get_label(self.image_list[index])
        
        # resize the image if resize first
        if self.resize_first:
            image, box = self._resize_image_first(image, box)
        
        # data augmentation
        image = self.crop_and_paste(image, box)
        image = self.crop_and_pad_safe(image, box)
        image = self.drop_block_safe(image, box)
        image, pos, ori, box = self.z_axis_rotation(image, pos, ori, box)
        image, pos, ori, box = self.persepctive_aug(image, pos, ori, box)
        image = self.albumentation_aug(image)

        # resize the image if not resize first
        if not self.resize_first:
            image, box = self._resize_image(image, box)

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

        return image_tensor, image, label




class SPEEDValDataset(SPEEDDataset):
    """
    The dataset class for SPEED val set.
    """
    def __init__(self, config: SPEEDConfig = SPEEDConfig()):
        super().__init__(config, "val")
    
    def __getitem__(self, index):
        image = self._get_image(self.image_list[index])
        pos, ori, box = self._get_label(self.image_list[index])

        # resize the image
        image, box = self._resize_image(image, box)

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

        return image_tensor, image, label


def get_dataloader(config: SPEEDConfig = SPEEDConfig()):
    data_loader = DataLoader if config.debug else MultiEpochsDataLoader
    # data_loader = DataLoader
    train_dataset = SPEEDTrainDataset(config)
    val_dataset = SPEEDValDataset(config)
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