"""
Test the postprocessing on the final trained model.
"""
import os

import numpy as np
import argparse
import pandas as pd
import toml
import torch
import torch.nn as nn
from tqdm import tqdm

from src.features.core import denoise, grow
from src.metrics import Metrics
from src.plain_dataloader import get_plain_dataset_loader
from src.unet import UNet


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

        outputs = np.argmax(np.array(outputs), axis=1).astype("float")
        for kernel_size_denoise in tqdm(
            list(range(1, 15)), desc="denoise", leave=False
        ):
            outputs_denoise = [
                denoise(img, kernel_size_denoise) for img in outputs
            ]  # You could do this slightly faster by splitting the loops
            for kernel_size_grow in tqdm(list(range(1, 10)), desc="grow", leave=False):
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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config path")
    parser.add_argument("checkpoint", help="checkpoint path")
    args = parser.parse_args()
    config = toml.load(args.config)
    # checkpoint_path = "experiment_20220407_151148/Green-checkpoint-100-of-100.pth"
    chkpt = torch.load(args.checkpoint, map_location=device)
    num_classes = 2

    net = UNet(2).to(device)
    net = nn.DataParallel(net)
    net.load_state_dict(chkpt["state_dict"])
    net.eval()

    # Optimize the postprocessing
    ds = "validation"
    ds_dir = os.path.join(config["dataset_path"], ds)
    loader = get_plain_dataset_loader(config["target_size"], 64, ds_dir)
    loader = get_plain_dataset_loader(config["target_size"], 64, ds_dir)
    tile_size = config["target_size"]
    opt = optimise_postprocessing(loader, num_classes, device, net)
    df_opt = pd.DataFrame(opt)
    print(df_opt)
    df_opt.to_csv("optimise_postprocessing.csv")
