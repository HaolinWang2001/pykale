"""
Dataset class to load data for few-shot learning problems under N-way-K-shot settings.
Author: Wenrui Fan
Email: winslow.fan@outlook.com
"""

import os
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class NWayKShotDataset(Dataset):
    """
    This Dataset class loads data for few-shot learning problems under N-way-K-shot settings.

    - N-way: The number of classes under a particular setting. The model is presented with samples from these N classes and needs to differentiate between them. For example, 3-way means the model has to distinguish between 3 different classes.

    - K-shot: The number of samples for each class in the support set. For example, in a 2-shot setting, two support samples are provided per class.

    In this class, __getitem__() returns a batch of images for one class. When defining the dataloaders, the batch size should be the number of classes (N-way). Therefore, __len__() returns the number of classes in the datasets.

    Note:
        The dataset should be organized as:

        - root
            - train
                - class_name 1
                    - xxx.png
                    - yyy.png
                    - ...
                - class_name 2
                    - xxx.png
                    - yyy.png
                    - ...
                - ...
            - val
                - class_name 1
                    - xxx.png
                    - yyy.png
                    - ...
                - class_name 2
                    - xxx.png
                    - yyy.png
                    - ...
                - ...
            - test
                - class_name 1
                    - xxx.png
                    - yyy.png
                    - ...
                - class_name 2
                    - xxx.png
                    - yyy.png
                    - ...
                - ...

    Args:
        path (string): The root directory of the data.
        mode (string): The mode of the type of dataset. It can be "train", "val", or "test". Default: "train".
        num_support_samples (int): Number of samples per class in the support set. It corresponds to K in the N-way-K-shot setting. Default: 5.
        num_query_samples (int): Number of samples per class in the query set. Default: 15.
        transform (callable, optional): Optional transform to be applied to images. Default: None.
    """

    def __init__(
        self,
        path: str,
        mode: str = "train",
        num_support_samples: int = 5,
        num_query_samples: int = 15,
        transform: Optional[Callable] = None,
    ):
        super(NWayKShotDataset, self).__init__()

        self.root = path
        self.num_support_samples = num_support_samples
        self.num_query_samples = num_query_samples
        self.mode = mode
        self.transform = transform
        self.images = []  # type: list
        self.labels = []  # type: list
        self._load_data()

    def __len__(self):
        # the length of the dataset is the number of classes in the dataset
        return len(self.classes)

    def __getitem__(self, idx):
        # sampling images from one class
        image_idx = self._get_idx(idx)
        images = self._sample_data(image_idx)
        assert isinstance(images, list)
        images = torch.stack(images)
        return images, idx

    def _get_idx(self, idx):
        # getting the indices of images for one class
        image_idx = np.random.choice(
            [i for (i, item) in enumerate(self.labels) if item == idx],
            self.num_support_samples + self.num_query_samples,
            replace=False
        )
        return image_idx

    def _sample_data(self, image_idx):
        # loading sampled images and applying transform 
        images = [self.transform(self.images[index]) if self.transform else self.images[index] for index in image_idx]
        return images

    def _load_data(self):
        # loading image list from the root directory
        data_path = os.path.join(self.root, self.mode)
        classes = os.listdir(data_path)
        classes.sort()
        self.classes = classes
        for i, c in enumerate(classes):
            c_dir = os.path.join(data_path, c)
            imgs = os.listdir(c_dir)
            for img in imgs:
                img_path = os.path.join(c_dir, img)
                self.images.append(Image.open(img_path).convert("RGB"))
                self.labels.append(i)
