from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, Normalize, Resize
from pathlib import Path

from src.transforms import (
    ConvertImageMode,
    ImageToTensor,
    JointCompose,
    JointTransform,
    MaskToTensor,
)

from src.datasets import NamedDataset, LabelledDataset

def get_plain_dataset_loader(target_size, batch_size, dataset_path):
    """
    A dataset loader for validation.
    """
    target_size = (target_size, target_size)
    # using imagenet mean and std for Normalization
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = JointCompose(
        [
            JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
            JointTransform(
                Resize(target_size, Image.BILINEAR), Resize(target_size, Image.NEAREST),
            ),
            JointTransform(CenterCrop(target_size), CenterCrop(target_size)),
            JointTransform(ImageToTensor(), MaskToTensor()),
            JointTransform(Normalize(mean=mean, std=std), None),
        ]
    )
    image_paths = list(Path(dataset_path).glob("*/*.png"))
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
    transform = JointCompose(
        [
            JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
            JointTransform(
                Resize(target_size, Image.BILINEAR), Resize(target_size, Image.NEAREST),
            ),
            JointTransform(CenterCrop(target_size), CenterCrop(target_size)),
            JointTransform(ImageToTensor(), MaskToTensor()),
            JointTransform(Normalize(mean=mean, std=std), None),
        ]
    )
    image_paths = list(Path(dataset_path).glob("*/*.png"))
    dataset = NamedDataset(image_paths=image_paths, joint_transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)

    return loader