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

    # We will iterate over the "best config" and its kfolds
    original_config = toml.load("config/best-predict-config.toml")
    kfold_config_paths = list(Path("results").glob("kfold_*/config.toml"))
    assert kfold_config_paths, "did kfold already get run?"
    print(f"{len(kfold_config_paths)=}")

    # Check the kfold files match the original config.
    for p in kfold_config_paths:
        print(p)
        config = toml.load(p)
        for key in original_config:
            if key in ("dataset_path", "checkpoint_path", "kfold"):
                continue
            if config[key] != original_config[key]:
                raise ValueError(
                    "Non matching kfold config"
                    + f"{p} {key} {config[key]} != {original_config[key]}"
                )

    loss_func = original_config["loss_func"]
    # weight = config["weight"]
    weight = [original_config["signal_fraction"], 1]
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
        for ds in ("training_s", "training_b", "validation", "testing"):
            if ds == "testing":
                ds_dir = Path(config["dataset_path"]).parent / "testing"
            else:
                ds_dir = Path(config["dataset_path"]) / ds
            print(ds_dir)
            loader = get_plain_dataset_loader(config["target_size"], batch_size, ds_dir)
            tile_size = config["target_size"]

            val = validate(loader, num_classes, device, net, criterion)
            # val = validate_offline(loader, num_classes, device, net)
            val["kfold"] = config["kfold"]
            val["ds"] = ds
            results.append(val)
            print(val)

    df = pd.DataFrame(results)
    df.to_csv("confusion_matrix.csv", index=False)
    print(df)
    print("wrote to confusion_matrix.csv")
