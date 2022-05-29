import argparse
import pandas as pd
import collections
import os
import sys

import numpy as np
import toml
import torch
import torch.nn as nn
from PIL import Image
from torch.nn import DataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, Normalize, Resize
from tqdm import tqdm

from src.datasets import SlippyMapTilesConcatenation
from src.extract import intersection
from src.features.core import denoise, grow
from src.metrics import Metrics
from src.predict import predict
from src.transforms import (
    ConvertImageMode,
    ImageToTensor,
    JointCompose,
    JointRandomHorizontalFlip,
    JointRandomRotation,
    JointTransform,
    MaskToTensor,
)

# from src.train import validate
from src.unet import UNet


def get_plain_dataset_loader(target_size, batch_size, dataset_path):
    target_size = (target_size, target_size)
    # using imagenet mean and std for Normalization
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = JointCompose(
        [
            JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
            JointTransform(
                Resize(target_size, Image.BILINEAR),
                Resize(target_size, Image.NEAREST),
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


def optimise_postprocessing(loader, num_classes, device, net):
    kernel_size_denoise = 15
    kernel_size_grow = 10
    num_samples = 0

    result = []

    with torch.no_grad():
        net.eval()

        outputs, masks_list = [], []
        for images, masks, tiles in tqdm(
            loader, desc="predict", unit="batch", ascii=True
        ):
            images = images.to(device)
            masks = masks.to(device)

            assert (
                images.size()[2:] == masks.size()[1:]
            ), "resolutions for images and masks are in sync"

            num_samples += int(images.size(0))
            outputs.append(net(images))
            masks_list.append(masks)

        outputs = np.concatenate(outputs, axis=0)
        masks = torch.Tensor(np.concatenate(masks_list, axis=0))

        outputs= np.argmax(np.array(outputs), axis=1).astype("float")
        for kernel_size_denoise in tqdm(list(range(1,15)), desc="denoise", leave=False):
            outputs_denoise = [
                denoise(img, kernel_size_denoise) for img in outputs
            ] # You could do this slightly faster by splitting the loops
            for kernel_size_grow in tqdm(list(range(1,10)), desc="grow", leave=False):
                metrics = Metrics(range(num_classes))
                outputs_postprocess = [
                    grow(img, kernel_size_grow) for img in outputs_denoise
                ]
                outputs_postprocess = torch.Tensor(outputs_postprocess)
                assert masks.shape == outputs_postprocess.shape
                for mask, output in zip(masks, outputs_postprocess):
                    metrics.add_binary(mask, output)

                result.append(
                    {
                        "miou": metrics.get_miou(),
                        "fg_iou": metrics.get_fg_iou(),
                        "mcc": metrics.get_mcc(),
                        "fp": metrics.fp,
                        "tp": metrics.tp,
                        "fn": metrics.fn,
                        "tn": metrics.tn,
                        "kernel_size_grow": kernel_size_grow,
                        "kernel_size_denoise": kernel_size_denoise,
                    }
                )
                breakpoint()

        return result


def validate(loader, num_classes, device, net, use_postprocess=False):
    kernel_size_denoise = 1
    kernel_size_grow = 9
    num_samples = 0

    metrics = Metrics(range(num_classes))
    metrics_postprocess = Metrics(range(num_classes))

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
            if use_postprocess:
                outputs_postprocess = np.argmax(np.array(outputs), axis=1).astype(
                    "float"
                )
                outputs_postprocess = [
                    denoise(img, kernel_size_denoise) for img in outputs_postprocess
                ]
                outputs_postprocess = [
                    grow(img, kernel_size_grow) for img in outputs_postprocess
                ]
                outputs_postprocess = torch.Tensor(outputs_postprocess)

            for mask, output in zip(masks, outputs):
                metrics.add(mask, output)

            if use_postprocess:
                for mask, output in zip(masks, outputs_postprocess):
                    metrics_postprocess.add_binary(mask, output)

        result = {
            "miou": metrics.get_miou(),
            "fg_iou": metrics.get_fg_iou(),
            "mcc": metrics.get_mcc(),
            "fp": metrics.fp,
            "tp": metrics.tp,
            "fn": metrics.fn,
            "tn": metrics.tn,
        }
        if use_postprocess:
            result_postprocess = {
                "miou": metrics_postprocess.get_miou(),
                "fg_iou": metrics_postprocess.get_fg_iou(),
                "mcc": metrics_postprocess.get_mcc(),
                "fp": metrics_postprocess.fp,
                "tp": metrics_postprocess.tp,
                "fn": metrics_postprocess.fn,
                "tn": metrics_postprocess.tn,
            }
        else:
            result_postprocess = {}

        return {"raw": result, "postprocess": result_postprocess}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_path = "experiment_20220407_151148/config.toml"
config = toml.load(config_path)
checkpoint_path = "experiment_20220407_151148/Green-checkpoint-100-of-100.pth"
chkpt = torch.load(checkpoint_path, map_location=device)
num_classes = 2

net = UNet(2).to(device)
net = nn.DataParallel(net)
net.load_state_dict(chkpt["state_dict"])
net.eval()

# Optimize the postprocessing
#ds = "validation"
#ds_dir = os.path.join("dataset", ds)
#loader = get_plain_dataset_loader(config["target_size"], 64, ds_dir)
#loader = get_plain_dataset_loader(config["target_size"], 64, ds_dir)
#tile_size = config["target_size"]
#opt = optimise_postprocessing(loader, num_classes, device, net)
#df_opt = pd.DataFrame(opt)
#print(df_opt)
#df_opt.to_csv("optimise_postprocessing.csv")



# Run on the datasets
results = []
results_postprocess = []
for ds in ("training", "validation", "evaluation"):
    ds_dir = os.path.join("dataset", ds)
    loader = get_plain_dataset_loader(config["target_size"], 64, ds_dir)
    tile_size = config["target_size"]

    val = validate(loader, num_classes, device, net, True)
    val["raw"]["ds"] = ds
    val["raw"]["postprocess"] = False
    val["postprocess"]["ds"] = ds
    val["postprocess"]["postprocess"] = True
    results.append(val["raw"])
    results_postprocess.append(val["postprocess"])



df = pd.DataFrame(results + results_postprocess)
df.to_csv("confusion_matrix.csv")
print(df)
