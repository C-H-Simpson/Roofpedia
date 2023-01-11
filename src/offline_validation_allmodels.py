"""
Get the confusion matrix from an already trained model.
"""
from pathlib import Path

# import numpy as np
import pandas as pd
import toml
import torch
import torch.nn as nn

from src.losses import CrossEntropyLoss2d, FocalLoss2d, LovaszLoss2d, mIoULoss2d
from src.plain_dataloader import get_plain_dataset_loader
from src.train import validate

# from src.train import validate
from src.unet import UNet

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Can increase batch size on a better GPU.
    # Doesn't need to be the same batch size as in training.
    batch_size = 8

    # We will iterate over all the available experiments.
    model_paths = list(Path("results").glob("experiment*/final_checkpoint.pth"))
    config_paths = [m.parent / "config.toml" for m in model_paths]
    results = []

    for config_p in config_paths:
        print(config_p)
        config = toml.load(config_p)

        loss_func = config["loss_func"]
        weight = [float(f) for f in config["weight"]]
        weight = torch.Tensor(weight)
        # select loss function, just set a default, or try to experiment
        if loss_func == "CrossEntropy":
            criterion = CrossEntropyLoss2d(weight=weight).to(device)
        elif loss_func == "mIoU":
            criterion = mIoULoss2d(weight=weight).to(device)
        elif loss_func == "Focal":
            criterion = FocalLoss2d(weight=weight).to(device)
        elif loss_func == "Lovasz":
            criterion = LovaszLoss2d().to(device)
        else:
            raise ValueError("Unknown Loss Function value !")

        chkpt_path = Path(config["checkpoint_path"]) / "final_checkpoint.pth"
        chkpt = torch.load(chkpt_path, map_location=device)
        num_classes = 2

        net = UNet(2).to(device)
        net = nn.DataParallel(net)
        net.load_state_dict(chkpt["state_dict"])
        net.eval()

        # Run on the datasets
        for ds in (
            "training_s", "training_b", "validation",
            "testing", "validation_alt", "testing_alt",
        ):
            if ds == "testing":
                ds_dir = Path(config["dataset_path"]).parent / "testing"
            elif ds == "testing_alt":
                ds_dir = Path(config["dataset_path"]).parent / "testing_alt"
            else:
                ds_dir = Path(config["dataset_path"]) / ds
            print(ds_dir)
            tile_size = config["target_size"]

            # Get validation data.
            loader = get_plain_dataset_loader(config["target_size"], batch_size, ds_dir)

            val = validate(loader, num_classes, device, net, criterion)
            # val = validate_offline(loader, num_classes, device, net)
            val["ds"] = ds
            config["config_path"] = config_p
            results.append({**val, **config})
            print(val)

    df = pd.DataFrame(results)
    df.to_csv("confusion_matrix_allmodels.csv", index=False)
    print(df)
    print("wrote to confusion_matrix_allmodels.csv")
