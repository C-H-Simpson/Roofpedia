"""PyTorch-compatible datasets.

Guaranteed to implement `__len__`, and `__getitem__`.

See: http://pytorch.org/docs/0.3.1/data.html
"""

import torch
from PIL import Image
from torch.utils.data import Dataset


class LabelledDataset(Dataset):
    """Return image/mask pairs"""

    def __init__(self, image_paths, joint_transform=None):
        self.img_paths = image_paths
        self.joint_transform = joint_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label_path = str(img_path).replace("images", "labels")
        image = [
            Image.open(img_path),
        ]  # transform expects and iterable
        label = Image.open(label_path)
        if self.joint_transform:
            image, label = self.joint_transform(image, label)
        return torch.cat(image, dim=0), label


class NamedDataset(Dataset):
    """Return images and their filenames"""

    def __init__(self, image_paths, joint_transform=None):
        self.img_paths = image_paths
        self.joint_transform = joint_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label_path = str(img_path).replace("images", "labels")
        image = [
            Image.open(img_path),
        ]  # transform expects an iterable
        label = Image.open(label_path)
        if self.joint_transform:
            image, label = self.joint_transform(image, label)
        x, y = int(img_path.parent.stem), int(img_path.stem)
        return torch.cat(image, dim=0), label, (x, y)
