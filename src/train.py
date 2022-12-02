from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, Normalize, Resize
from tqdm import tqdm

from src.datasets import SlippyMapTilesConcatenation
from src.metrics import Metrics
from src.resampling_dataloader import BackgroundResamplingLoader
from src.transforms import (
    ConvertImageMode,
    ImageToTensor,
    JointCompose,
    JointRandomHorizontalFlip,
    JointRandomRotation,
    JointTransform,
    MaskToTensor,
)


def get_dataset_loaders(
    target_size, batch_size, dataset_path, training_signal_fraction, transform=None
):
    target_size = (target_size, target_size)
    dataset_path = Path(dataset_path)
    # using imagenet mean and std for Normalization
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if transform is None:
        transform = JointCompose(
            [
                JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
                JointTransform(
                    Resize(target_size, Image.BILINEAR),
                    Resize(target_size, Image.NEAREST),
                ),
                JointTransform(CenterCrop(target_size), CenterCrop(target_size)),
                JointRandomHorizontalFlip(0.5),
                JointRandomRotation(0.5, 90),
                JointRandomRotation(0.5, 90),
                JointRandomRotation(0.5, 90),
                JointTransform(ImageToTensor(), MaskToTensor()),
                JointTransform(Normalize(mean=mean, std=std), None),
            ]
        )
    val_transform = JointCompose(
        [
            JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
            JointTransform(
                Resize(target_size, Image.BILINEAR),
                Resize(target_size, Image.NEAREST),
            ),
            JointTransform(CenterCrop(target_size), CenterCrop(target_size)),
            JointRandomHorizontalFlip(0.5),
            JointRandomRotation(0.5, 90),
            JointRandomRotation(0.5, 90),
            JointRandomRotation(0.5, 90),
            JointTransform(ImageToTensor(), MaskToTensor()),
            JointTransform(Normalize(mean=mean, std=std), None),
        ]
    )
    train_dataset = SlippyMapTilesConcatenation(
        [str(dataset_path / "training_s" / "images")],
        str(dataset_path / "training_s" / "labels"),
        transform,
    )

    train_bg_dataset = SlippyMapTilesConcatenation(
        [str(dataset_path / "training_b" / "images")],
        str(dataset_path / "training_b" / "labels"),
        transform,
    )

    val_dataset = SlippyMapTilesConcatenation(
        [str(dataset_path / "validation" / "images")],
        str(dataset_path / "validation" / "labels"),
        val_transform,
    )

    assert len(train_bg_dataset) > 0, "at least one tile in training background dataset"
    assert len(train_dataset) > 0, "at least one tile in training dataset"
    assert len(val_dataset) > 0, "at least one tile in validation dataset"
    print(
        f"Dataset sizes: len(train_dataset)={len(train_dataset)}, len(val_dataset)={len(val_dataset)}"
    )

    train_loader = DataLoader(
        BackgroundResamplingLoader(
            train_dataset, train_bg_dataset, training_signal_fraction
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    return train_loader, val_loader


def train(loader, num_classes, device, net, optimizer, criterion):
    num_samples = 0
    running_loss = 0

    # always two classes in our case
    metrics = Metrics(range(num_classes))
    # initialized model
    net.train()

    # training loop
    for images, masks, tiles in tqdm(loader, desc="Train", unit="batch", ascii=True):
        num_samples += int(images.size(0))

        images = images.to(device)
        masks = masks.to(device)

        assert (
            images.size()[2:] == masks.size()[1:]
        ), "resolutions for images and masks are in sync"

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        for mask, output in zip(masks, outputs):
            prediction = output.detach()
            metrics.add(mask, prediction)

    results = {
        "loss": running_loss / num_samples,
        "miou": metrics.get_miou(),
        "fg_iou": metrics.get_fg_iou(),
        "mcc": metrics.get_mcc(),
        "fp": metrics.fp,
        "tp": metrics.tp,
        "fn": metrics.fn,
        "tn": metrics.tn,
        "tp+fn": metrics.tp + metrics.fn,
        "tp+fn+fp+tn": metrics.tp + metrics.fn + metrics.fp + metrics.tn,
        "M": metrics.M,
    }
    try:
        results["f1"] = metrics.tp / (metrics.tp + 0.5 * (metrics.fp + metrics.fn))
    except ZeroDivisionError:
        results["f1"] = float("NAN")
    return results


def validate(loader, num_classes, device, net, criterion):
    num_samples = 0
    running_loss = 0

    metrics = Metrics(range(num_classes))

    with torch.no_grad():
        net.eval()

        for images, masks, tiles in tqdm(
            loader, desc="Validate", unit="batch", ascii=True
        ):
            images = images.to(device)
            masks = masks.to(device)

            assert (
                images.size()[2:] == masks.size()[1:]
            ), "resolutions for images and masks are in sync"

            num_samples += int(images.size(0))
            outputs = net(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            for mask, output in zip(masks, outputs):
                metrics.add(mask, output)

        results = {
            "loss": running_loss / num_samples,
            "miou": metrics.get_miou(),
            "fg_iou": metrics.get_fg_iou(),
            "mcc": metrics.get_mcc(),
            "fp": metrics.fp,
            "tp": metrics.tp,
            "fn": metrics.fn,
            "tn": metrics.tn,
            "tp+fn": metrics.tp + metrics.fn,
            "tp+fn+fp+tn": metrics.tp + metrics.fn + metrics.fp + metrics.tn,
            "M": metrics.M,
        }
        try:
            results["f1"] = metrics.tp / (metrics.tp + 0.5 * (metrics.fp + metrics.fn))
        except ZeroDivisionError:
            results["f1"] = float("NAN")
        return results
