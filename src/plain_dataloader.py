import os

from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, Normalize, Resize

from src.datasets import SlippyMapTilesConcatenation
from src.transforms import (
    ConvertImageMode,
    ImageToTensor,
    JointCompose,
    JointTransform,
    MaskToTensor,
)


def get_plain_dataset_loader(target_size, batch_size, dataset_path):
    """
    A dataset loader for prediction / validation.
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
    dataset = SlippyMapTilesConcatenation(
        [os.path.join(dataset_path, "images")],
        os.path.join(dataset_path, "labels"),
        transform,
    )
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)

    return loader
