"""
Repeat training across multiple folds, and produce performance statistics.
"""

import datetime
from pathlib import Path

import toml

from experiment import run_training
from src.augmentations import get_transforms
from src.plain_dataloader import LabelledDataset
from src.train import validate
from torch.utils.data import DataLoader

if __name__ == "__main__":
    config = toml.load("config/best-predict-config.toml")

    num_classes = 2
    lr = config["lr"]
    loss_func = config["loss_func"]
    num_epochs = config["num_epochs"]
    target_size = config["target_size"]
    batch_size = config["batch_size"]
    # dataset_path = config["dataset_path"]  # Changes below
    checkpoint_path = Path(config["checkpoint_path"])
    target_type = config["target_type"]
    freeze_pretrained = config["freeze_pretrained"]
    signal_fraction = config["signal_fraction"]
    weight = [float(w) for w in config["weight"]]
    transform_name = config["transform_name"]
    # Training a model from scratch
    config["model_path"] = ""
    model_path = ""
    augs = get_transforms(target_size)
    original_k = config["kfold"]

    # "Fold 0" is the test data in this code,
    # and fold 1 was used to train the "original" model in experiment.py
    # so there will only be three iterations with five folds.
    k_folds = 5
    for k in range(1, k_folds):
        if k == original_k:
            continue
        print(f"Starting fold {k} of {k_folds}")
        config["kfold"] = k
        # make dir for checkpoint - will get moved
        checkpoint_path = f"results/kfold_{k}_" + datetime.datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        config["checkpoint_path"] = checkpoint_path
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.mkdir(exist_ok=True)
        dataset_path = f"dataset/k{k}"
        config["dataset_path"] = dataset_path
        alt_validation_path = f"dataset/k{k}/validation_alt"
        config["alt_validation_path"] = alt_validation_path
        # Write the testing config to file
        with open(checkpoint_path / "config.toml", "w") as f:
            toml.dump(config, f)

        run_training(
            augs=augs,
            batch_size=batch_size,
            checkpoint_path=checkpoint_path,
            dataset_path=dataset_path,
            freeze_pretrained=freeze_pretrained,
            loss_func=loss_func,
            lr=lr,
            model_path=model_path,
            num_classes=num_classes,
            num_epochs=num_epochs,
            signal_fraction=signal_fraction,
            target_size=target_size,
            transform_name=transform_name,
            weight=weight,
            alt_validation_path=alt_validation_path,
            resampling_method=config["resampling_method"],
            early_stopping_window=config["early_stopping_window"],
            focal_gamma=config["focal_gamma"]
        )
