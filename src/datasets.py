"""PyTorch-compatible datasets.

Guaranteed to implement `__len__`, and `__getitem__`.

See: http://pytorch.org/docs/0.3.1/data.html
"""

from torch.utils.data import Dataset
from PIL import Image

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
        image = Image.open(img_path)
        label = Image.open(label_path)
        if self.joint_transform:
            image = self.joint_transform(image, label)
        return image, label

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
        image = Image.open(img_path)
        label = Image.opne(label_path)
        if self.joint_transform:
            image = self.joint_transform(image, label)
        return image, label, img_path
