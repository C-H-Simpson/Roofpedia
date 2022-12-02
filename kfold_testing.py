"""
Repeat training across multiple folds, and produce performance statistics.
"""

import collections
import datetime
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import toml
import torch
from torch.nn import DataParallel
from torch.optim import Adam

from src.augmentations import get_transforms
from src.losses import CrossEntropyLoss2d, FocalLoss2d, LovaszLoss2d, mIoULoss2d
from src.train import get_dataset_loaders, train, validate
from src.unet import UNet
from src.utils import plot
from src.plain_dataloader import get_plain_dataset_loader


def run_training():
    device = torch.device("cuda")

    if not torch.cuda.is_available():
        sys.exit("Error: CUDA requested but not available")

    # weighted values for loss functions
    # add a helper to return weights seamlessly
    # The weights should actually be based on the proportions in the loader...
    # This is currently only correct if there is no under/over sampling.
    # Lovasz does not use weighting.
    if loss_func != "Lovasz":
        weight = torch.Tensor([signal_fraction, 1])
    else:
        weight = None

    # loading Model
    net = UNet(num_classes, freeze_pretrained=freeze_pretrained)
    net = DataParallel(net)
    net = net.to(device)

    # define optimizer
    optimizer = Adam(net.parameters(), lr=lr)

    # resume training
    if model_path:
        chkpt = torch.load(model_path, map_location=device)
        net.load_state_dict(chkpt["state_dict"])
        optimizer.load_state_dict(chkpt["optimizer"])

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
        sys.exit("Error: Unknown Loss Function value !")

    # loading data
    train_loader, val_loader = get_dataset_loaders(
        target_size, batch_size, dataset_path, signal_fraction, augs[transform_name]
    )
    eval_loader = get_plain_dataset_loader(
        target_size, batch_size, Path(dataset_path).parent / "testing"
    )
    history = collections.defaultdict(list)

    # training loop
    for epoch in range(0, num_epochs):

        print("Epoch: " + str(epoch + 1))
        train_hist = train(train_loader, num_classes, device, net, optimizer, criterion)
        val_hist = validate(val_loader, num_classes, device, net, criterion)
        eval_hist = validate(eval_loader, num_classes, device, net, criterion)

        print(
            "Train stats:",
            ", ".join([f"{key}: {train_hist[key]:.3e}" for key in train_hist]),
        )
        print(
            "Validation stats:",
            ", ".join([f"{key}: {val_hist[key]:.3e}" for key in val_hist]),
        )

        for label, hist in (
            ("train", train_hist),
            ("val", val_hist),
            ("eval", eval_hist),
        ):
            for key, value in hist.items():
                history[label + key].append(value)

        with open(checkpoint_path / "history.json", "w") as f:
            json.dump(history, f)

        if (epoch + 1) % 5 == 0:
            # plotter use history values, no need for log
            visual = "history-{:05d}-of-{:05d}.png".format(epoch + 1, num_epochs)
            plot(checkpoint_path / visual, history)

        if epoch > 10:
            if np.mean(history["val loss"][-11:-6]) < np.mean(history["val loss"][-6:]):
                break

    # Save the model
    checkpoint = target_type + "-checkpoint-{:03d}-of-{:03d}.pth".format(
        epoch + 1, num_epochs
    )
    states = {
        "epoch": epoch + 1,
        "state_dict": net.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(states, checkpoint_path / checkpoint)


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
    transform_name = config["transform"]
    # Training a model from scratch
    config["model_path"] = ""
    model_path = ""
    augs = get_transforms(target_size)

    # Fold 0 is the test data in this code, so there will be 4 iterations with 5 folds.
    k_folds = 5
    for k in range(1, k_folds):
        print(f"Starting fold {k} of {k_folds}")
        config["kfold"] = k
        # make dir for checkpoint - will get moved
        checkpoint_path.mkdir(exist_ok=True)
        dataset_path = f"dataset/k{k}"
        fname = f"results/kfold_{k}_" + datetime.datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        config["checkpoint_path"] = fname
        # Write the testing config to file
        with open(checkpoint_path / "config.toml", "w") as f:
            toml.dump(config, f)

        # NB this routine is not the same as in experiment.py
        run_training()

        # Move the config and results to a new directory
        shutil.move(checkpoint_path, fname)
