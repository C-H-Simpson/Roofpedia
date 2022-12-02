"""
Get the confusion matrix from an already trained model.
"""
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import toml
import torch
import torch.nn as nn
from tqdm import tqdm

from src.features.core import denoise, grow
from src.metrics import Metrics
from src.plain_dataloader import get_plain_dataset_loader

# from src.train import validate
from src.unet import UNet


def validate_offline(loader, num_classes, device, net, use_postprocess=False):
    """This version of the validate routine can be used to test postprocess.
    Also, can take different loaders as input."""
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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We will iterate over the "best config" and its kfolds
    original_config = toml.load("config/best-predict-config.toml")
    kfold_config_paths = Path("results").glob("kfold*")

    # Check the kfold files match the original config.
    for p in kfold_config_paths:
        config = toml.load(p / "config.toml")
        for key in original_config:
            if key in ("dataset_path", "checkpoint_path", "kfold"):
                continue
            if config[key] != original_config[key]:
                raise ValueError(
                    f"Non matching kfold config {p} {key} {config[key]} != {original_config[key]}"
                )

        print(f"{config['kfold']=}")

    results = []
    results_postprocess = []

    # Iterate through k folds and do offline validation.
    for config in [original_config] + [
        toml.load(p / "config.toml") for p in kfold_config_paths
    ]:
        chkpt_path = Path(config["checkpoint_path"]) / "final_checkpoint.pth"
        chkpt = torch.load(chkpt_path, map_location=device)
        num_classes = 2

        net = UNet(2).to(device)
        net = nn.DataParallel(net)
        net.load_state_dict(chkpt["state_dict"])
        net.eval()

        # Run on the datasets
        for ds in ("training", "validation", "evaluation"):
            ds_dir = os.path.join(config["dataset_path"], ds)
            loader = get_plain_dataset_loader(config["target_size"], 64, ds_dir)
            tile_size = config["target_size"]

            val = validate_offline(loader, num_classes, device, net, True)
            val["raw"]["ds"] = ds
            val["raw"]["postprocess"] = False
            val["postprocess"]["ds"] = ds
            val["postprocess"]["postprocess"] = True
            results.append(val["raw"])
            results_postprocess.append(val["postprocess"])
        break

    df = pd.DataFrame(results + results_postprocess)
    # df.to_csv("confusion_matrix.csv")
    print(df)
    print("wrote to confusion_matrix.csv")
