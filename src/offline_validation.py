"""
Get the confusion matrix from an already trained model.
"""
import os
from pathlib import Path

# import numpy as np
import pandas as pd
import toml
import torch
import torch.nn as nn
from tqdm import tqdm

# from src.features.core import denoise, grow
from src.metrics import Metrics
from src.plain_dataloader import get_plain_dataset_loader

# from src.train import validate
from src.unet import UNet


def validate_offline(loader, num_classes, device, net):
    """This version of the validate routine can be used to test postprocess.
    Also, can take different loaders as input."""
    num_samples = 0

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

            for mask, output in zip(masks, outputs):
                metrics.add(mask, output)

        result = {
            "miou": metrics.get_miou(),
            "fg_iou": metrics.get_fg_iou(),
            "mcc": metrics.get_mcc(),
            "fp": metrics.fp,
            "tp": metrics.tp,
            "fn": metrics.fn,
            "tn": metrics.tn,
            "tp+fn": metrics.tp + metrics.fn,
        }
        try:
            result["f1"] = metrics.tp / (metrics.tp + 0.5 * (metrics.fp + metrics.fn))
        except ZeroDivisionError:
            result["f1"] = float("NAN")
        return result


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Can increase batch size on a better GPU.
    # Doesn't need to be the same batch size as in training.
    batch_size = 8

    # We will iterate over the "best config" and its kfolds
    original_config = toml.load("config/best-predict-config.toml")
    kfold_config_paths = list(Path("results").glob("kfold*/config.toml"))

    # Check the kfold files match the original config.
    for p in kfold_config_paths:
        config = toml.load(p)
        for key in original_config:
            if key in ("dataset_path", "checkpoint_path", "kfold"):
                continue
            if config[key] != original_config[key]:
                raise ValueError(
                    "Non matching kfold config"
                    + f"{p} {key} {config[key]} != {original_config[key]}"
                )

    # Iterate through k folds and do offline validation.
    results = []
    kfold_config_paths.append("config/best-predict-config.toml")
    for config in kfold_config_paths:
        print(config)
        config = toml.load(config)
        chkpt_path = Path(config["checkpoint_path"]) / "final_checkpoint.pth"
        chkpt = torch.load(chkpt_path, map_location=device)
        num_classes = 2

        net = UNet(2).to(device)
        net = nn.DataParallel(net)
        net.load_state_dict(chkpt["state_dict"])
        net.eval()

        # Run on the datasets
        for ds in ("training_s", "training_b", "validation", "evaluation"):
            if ds == "evaluation":
                ds_dir = Path(config["dataset_path"]).parent / "testing"
            else:
                ds_dir = Path(config["dataset_path"]) / ds
            print(ds_dir)
            loader = get_plain_dataset_loader(config["target_size"], batch_size, ds_dir)
            tile_size = config["target_size"]

            val = validate_offline(loader, num_classes, device, net)
            val["kfold"] = config["kfold"]
            val["ds"] = ds
            results.append(val)

    df = pd.DataFrame(results)
    df.to_csv("confusion_matrix.csv")
    print(df)
    print("wrote to confusion_matrix.csv")
