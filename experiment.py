"""
Run experiments to find a good training configuration.
"""
# %%
import collections
import datetime
import json

# import shutil
import sys
from pathlib import Path

import numpy as np
import toml
import torch
from torch.nn import DataParallel
from torch.optim import Adam

from dataset_stats import count_signal_pixels
from src.augmentations import get_transforms
from src.losses import (
    CrossEntropyLoss2d,
    FocalLoss2d,
    LovaszLoss2d,
    mIoULoss2d,
    FLoss2d,
)
from src.plain_dataloader import get_plain_dataset_loader
from src.train import get_dataset_loaders, train, validate
from src.unet import UNet
from src.utils import plot


def run_training(
    *,
    alt_validation_path,
    augs,
    batch_size,
    checkpoint_path,
    dataset_path,
    freeze_pretrained,
    loss_func,
    lr,
    model_path,
    num_classes,
    num_epochs,
    signal_fraction,
    target_size,
    transform_name,
    weight,
    resampling_method,
    early_stopping_window,
    focal_gamma=None,
):
    device = torch.device("cuda")

    if not torch.cuda.is_available():
        raise ValueError("Error: CUDA requested but not available")

    # weighted values for loss functions
    # add a helper to return weight seamlessly

    # The weight should actually be based on the proportions in the loader...
    weight = torch.Tensor(weight)

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
        criterion = FocalLoss2d(weight=weight, gamma=focal_gamma).to(device)
    elif loss_func == "Lovasz":
        criterion = LovaszLoss2d().to(device)
    elif loss_func == "F1":
        criterion = FLoss2d().to(device)
    else:
        raise ValueError("Error: Unknown Loss Function value !")

    # loading data
    train_loader, val_loader = get_dataset_loaders(
        target_size,
        batch_size,
        dataset_path,
        signal_fraction,
        augs[transform_name],
        resampling_method=resampling_method,
        base_transform=augs["no_augs_A"],
    )
    alt_validation_path = Path(alt_validation_path)
    if alt_validation_path.is_dir():
        alt_val_loader = get_plain_dataset_loader(
            target_size, batch_size, alt_validation_path, base_transform="no_augs_A"
        )
    else:
        # print("Alternative validation data not available", alt_validation_path)
        alt_val_loader = None
        # Comment this error out to make alternative validation data non-compulsory.
        # raise FileNotFoundError(
        #     f"Alternative validation data not available {alt_validation_path}"
        # )
    if not len(alt_val_loader):
        alt_val_loader = None

    history = collections.defaultdict(list)

    # training loop
    for epoch in range(0, num_epochs):

        print("Epoch: " + str(epoch + 1))
        train_hist = train(train_loader, num_classes, device, net, optimizer, criterion)

        val_hist = validate(val_loader, num_classes, device, net, criterion)

        if alt_val_loader is not None:
            alt_val_hist = validate(alt_val_loader, num_classes, device, net, criterion)

        print(
            "Train stats:",
            ", ".join([f"{key}: {train_hist[key]:.3e}" for key in train_hist]),
        )

        print(
            "Validation stats:",
            ", ".join([f"{key}: {val_hist[key]:.3e}" for key in val_hist]),
        )
        if alt_val_loader is not None:
            print(
                "Alt validation stats:",
                ", ".join([f"{key}: {alt_val_hist[key]:.3e}" for key in alt_val_hist]),
            )

        for key, value in train_hist.items():
            history["train " + key].append(value)

        for key, value in val_hist.items():
            history["val " + key].append(value)

        if alt_val_loader is not None:
            for key, value in alt_val_hist.items():
                history["alt val " + key].append(value)

        with open(Path(checkpoint_path) / "history.json", "w") as f:
            json.dump(history, f)

        if (epoch + 1) % 5 == 0:
            # plotter use history values, no need for log
            visual = "history-{:05d}-of-{:05d}.png".format(epoch + 1, num_epochs)
            plot(Path(checkpoint_path) / visual, history)

        if epoch > early_stopping_window * 2:
            w1 = early_stopping_window + 1
            w2 = early_stopping_window * 2 + 1
            f1_before = np.mean(history["val f1"][-w2:-w1])
            f1_now = np.mean(history["val f1"][-w1:])
            if not (f1_now > f1_before):
                print("Early stopping")
                break

    # Save the model
    states = {
        "epoch": epoch + 1,
        "state_dict": net.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(states, Path(checkpoint_path) / "final_checkpoint.pth")

    # eval_loader = get_plain_dataset_loader(
    #     target_size, batch_size, Path(dataset_path).parent / "testing"
    # )
    return history["val f1"][-1]

    # eval_hist = validate(eval_loader, num_classes, device, net, criterion)


def experiment(config):
    # Work out weighting for loss functions
    pixel_weights = count_signal_pixels(
        (Path(config["dataset_path"]).glob("training_s/labels/*/*png"))
    )
    if config["signal_fraction"] != -1:
        n_samples = (
            pixel_weights["n_signal_tiles"]
            * (config["tile_size"] ** 2)
            / config["signal_fraction"]
        )
    else:
        n_samples = pixel_weights["n_tiles"] * 256 * 256
    n_samplesj = pixel_weights["n_signal_pixels"]
    pos_weight = n_samples / (
        config["num_classes"] * n_samplesj
    )  # Inverse frequency weighting.
    neg_weight = n_samples / (config["num_classes"] * (n_samples - n_samplesj))
    weight = [
        neg_weight,
        pos_weight,
    ]
    print(weight)
    config["weight"] = weight
    assert pos_weight > neg_weight

    # Training a model from scratch
    config["model_path"] = ""
    # make dir for checkpoint
    fname = "results/experiment_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config["checkpoint_path"] = fname

    # Make output directory
    checkpoint_path = Path(fname)
    checkpoint_path.mkdir(exist_ok=False)

    # Read from config
    augs = get_transforms(config["target_size"])

    # Write the testing config to file
    with open(Path(checkpoint_path) / "config.toml", "w") as f:
        f.write(toml.dumps(config))

    return run_training(
        alt_validation_path=config["alt_validation_path"],
        augs=augs,
        batch_size=config["batch_size"],
        checkpoint_path=config["checkpoint_path"],
        dataset_path=config["dataset_path"],
        freeze_pretrained=config["freeze_pretrained"],
        loss_func=config["loss_func"],
        lr=config["lr"],
        model_path=config["model_path"],
        num_classes=config["num_classes"],
        num_epochs=config["num_epochs"],
        signal_fraction=config["signal_fraction"],
        target_size=config["target_size"],
        transform_name=config["transform_name"],
        weight=config["weight"],
        focal_gamma=config["focal_gamma"],
        resampling_method=config["resampling_method"],
        early_stopping_window=config["early_stopping_window"],
    )


if __name__ == "__main__":
    best_config = toml.load("config/best-predict-config.toml")

    # Freeze the pretrained layers for the optimisation step.
    best_config["freeze_pretrained"] = True
    best_config["early_stopping_window"] = 5

    print("Experiment set 1: Does resampling the background help?")
    config = best_config
    config["transform_name"] = "no_augs_A"
    for resampling_method in ("background", "signal"):
        for signal_fraction in (1, 0.8, 0.6, 0.4, 0.2):
            config["resampling_method"] = "background"
            config["signal_fraction"] = signal_fraction
            for lr_factor in (0.1, 0.01):
                print(f"{signal_fraction=}, {lr_factor=}")
                config["lr"] = lr_base * lr_factor
                f1 = experiment(config)
                if f1 > best_f1:
                    best_f1 = f1
                    best_config = config
    print(f"{best_config['signal_fraction']=}, {best_config['resampling_method']=}")

    print("Experiment set 2: Is there a difference between the loss functions?")
    config = best_config
    config["signal_fraction"] = 0.1
    config["transform_name"] = "no_augs_A"
    for loss_func in ("F1", "Focal", "mIoU", "CrossEntropy"):
        config["loss_func"] = loss_func
        focal_gamma_loop = (2, 3, 4) if loss_func == "Focal" else (None,)
        for focal_gamma in focal_gamma_loop:
            config["focal_gamma"] = focal_gamma
            for lr_factor in (0.01,):
                config["lr"] = lr_base * lr_factor
                print(f"{loss_func=}, {focal_gamma=}, {lr_factor=}")
                f1 = experiment(config)
                if f1 > best_f1:
                    best_f1 = f1
                    best_config = config
    print(f"{best_config['loss_func']=}")

    print("Experiment set 3: Do the augmentations help?")
    config = best_config
    augs = get_transforms(best_config["target_size"])
    lr_base = best_config["lr"]
    for transform_name in (
        "no_augs_A",
        "flip_rotate_A",
        "non_spatial_E",
    ):
        config["transform_name"] = transform_name
        for lr_factor in (10, 1, 0.1, 0.01):
            config["lr"] = lr_base * lr_factor
            print(f"{transform_name=}, {lr_factor=}")
            f1 = experiment(config)
            if f1 > best_f1:
                best_f1 = f1
                best_config = config
    print(f"{best_config['transform_name']=}")
