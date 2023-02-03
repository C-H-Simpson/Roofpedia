"""PyTorch-compatible datasets.

Guaranteed to implement `__len__`, and `__getitem__`.

See: http://pytorch.org/docs/0.3.1/data.html
"""

import torch
from PIL import Image
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from src.augmentations import get_transforms

from albumentations.pytorch.transforms import ToTensorV2


class LabelledDataset(Dataset):
    """Return image/mask pairs"""

    def __init__(self, image_paths, joint_transform=None):
        self.img_paths = image_paths
        self.joint_transform = joint_transform
        if joint_transform == None:
            self.joint_transform = get_transforms()["no_augs_A"]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label_path = str(img_path).replace("images", "labels")
        # image = [
        #     # transforms.ToTensor()(Image.open(img_path)).unsqueeze_(0)
        #     # np.array(Image.open(img_path)),
        # ]  # transform expects and iterable
        # label = transforms.ToTensor()(Image.open(label_path)).unsqueeze_(0)
        image = cv.imread(str(img_path))
        label = (cv.imread(str(label_path), cv.IMREAD_GRAYSCALE) > 200).astype("int64")
        # label = (cv.imread(str(label_path))[:,:,0] > 0) # should be a 1 channel image
        # label = np.array(Image.open(label_path))
        # image = np.array(image)
        # breakpoint()
        out = self.joint_transform(image=image, mask=label)
        image, label = out["image"], out["mask"]
        return (
            torch.cat(
                [
                    image,
                ],
                dim=0,
            ),
            label,
        )


class NamedDataset(Dataset):
    """Return images and their filenames"""

    def __init__(self, image_paths, joint_transform=None):
        self.img_paths = image_paths
        self.joint_transform = joint_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        # label_path = str(img_path).replace("images", "labels")
        image = cv.imread(str(img_path))
        # label = (cv.imread(str(label_path), cv.IMREAD_GRAYSCALE) > 200).astype("int64")
        x, y = int(img_path.parent.stem), int(img_path.stem)
        out = self.joint_transform(image=image)
        image = out["image"]
        return (
            torch.cat(
                [
                    image,
                ],
                dim=0,
            ),
            (x, y),
        )
