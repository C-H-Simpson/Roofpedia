from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, Normalize, Resize
from tqdm import tqdm

from src.metrics import Metrics
from src.plain_dataloader import LabelledDataset
from src.resampling_dataloader import BackgroundResamplingLoader, SignalResamplingLoader
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
    target_size, batch_size, dataset_path, training_signal_fraction, transform=None,
    resampling_method="background"
):
    print(f"{resampling_method=}")
    target_size = (target_size, target_size)
    dataset_path = Path(dataset_path)
    # using imagenet mean and std for Normalization
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_s_image_paths = list(dataset_path.glob("training_s/images/*/*.png"))
    train_s_dataset = LabelledDataset(
        image_paths=train_s_image_paths, joint_transform=transform
    )

    train_b_image_paths = list(dataset_path.glob("training_b/images/*/*.png"))
    train_b_dataset = LabelledDataset(
        image_paths=train_b_image_paths, joint_transform=transform
    )

    val_image_paths = list(dataset_path.glob("validation/images/*/*.png"))
    val_dataset = LabelledDataset(
        image_paths=val_image_paths, joint_transform=None
    )

    assert len(train_b_dataset) > 0, "at least one tile in training background dataset"
    assert len(train_s_dataset) > 0, "at least one tile in training dataset"
    assert len(val_dataset) > 0, "at least one tile in validation dataset"
    print(
        f"Dataset sizes: len(train_s_dataset)={len(train_s_dataset)}, "
        + f"len(val_dataset)={len(val_dataset)}"
    )

    if resampling_method=="background":
        train_loader = DataLoader(
            BackgroundResamplingLoader(
                train_s_dataset, train_b_dataset, training_signal_fraction
            ),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
    elif resampling_method=="signal":
        train_loader = DataLoader(
            SignalResamplingLoader(
                train_s_dataset, train_b_dataset, training_signal_fraction
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
    for images, masks in tqdm(loader, desc="Train", unit="batch", ascii=True):
        num_samples += int(images.size(0))

        images = images.to(device)
        masks = masks.to(device)

        # breakpoint()
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
    try:
        results["precision"] = metrics.tp / (metrics.tp + metrics.fp)
    except ZeroDivisionError:
        results["precision"] = float("NAN")
    try:
        results["recall"] = metrics.tp / (metrics.tp + metrics.fn)
    except ZeroDivisionError:
        results["recall"] = float("NAN")
    return results


def validate(loader, num_classes, device, net, criterion):
    num_samples = 0
    running_loss = 0

    metrics = Metrics(range(num_classes))

    with torch.no_grad():
        net.eval()

        for images, masks in tqdm(loader, desc="Validate", unit="batch", ascii=True):
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
        try:
            results["precision"] = metrics.tp / (metrics.tp + metrics.fp)
        except ZeroDivisionError:
            results["precision"] = float("NAN")
        try:
            results["recall"] = metrics.tp / (metrics.tp + metrics.fn)
        except ZeroDivisionError:
            results["recall"] = float("NAN")
        return results
