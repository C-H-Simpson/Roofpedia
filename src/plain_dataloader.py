from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, Normalize, Resize

from src.datasets import LabelledDataset, NamedDataset
from src.transforms import (
    ConvertImageMode,
    ImageToTensor,
    JointCompose,
    JointTransform,
    MaskToTensor,
)
from src.augmentations import get_transforms


def get_plain_dataset_loader(target_size, batch_size, dataset_path):
    """
    A dataset loader for validation.
    """
    target_size = (target_size, target_size)
    # using imagenet mean and std for Normalization
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    augs = get_transforms(target_size)
    transform = augs["no_augs_A"]
    image_paths = list(Path(dataset_path).glob("images/*/*.png"))
    dataset = LabelledDataset(image_paths=image_paths, joint_transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)

    return loader


def get_named_dataset_loader(target_size, batch_size, dataset_path):
    """
    A dataset loader for prediction. Returns filenames.
    """
    target_size = (target_size, target_size)
    # using imagenet mean and std for Normalization
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    augs = get_transforms(target_size)
    transform = augs["no_augs_A"]
    image_paths = list(Path(dataset_path).glob("images/*/*.png"))
    dataset = NamedDataset(image_paths=image_paths, joint_transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)

    return loader
