"""
Run experiments to find a good training configuration.
"""
import collections
import datetime
import json
import os
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


def run_training(
    *,
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
):
    device = torch.device("cuda")

    if not torch.cuda.is_available():
        sys.exit("Error: CUDA requested but not available")

    # weighted values for loss functions
    # add a helper to return weight seamlessly

    # The weight should actually be based on the proportions in the loader...
    # This is currently only correct if there is no under/over sampling.
    # Lovasz does not use weighting.
    if loss_func != "Lovasz":
        weight = torch.Tensor(weight)
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
    history = collections.defaultdict(list)

    # training loop
    for epoch in range(0, num_epochs):

        print("Epoch: " + str(epoch + 1))
        train_hist = train(train_loader, num_classes, device, net, optimizer, criterion)

        val_hist = validate(val_loader, num_classes, device, net, criterion)

        print(
            "Train stats:",
            ", ".join([f"{key}: {train_hist[key]:.3e}" for key in train_hist]),
        )

        print(
            "Validation stats:",
            ", ".join([f"{key}: {val_hist[key]:.3e}" for key in val_hist]),
        )

        for key, value in train_hist.items():
            history["train " + key].append(value)

        for key, value in val_hist.items():
            history["val " + key].append(value)

        with open(checkpoint_path / "history.json", "w") as f:
            json.dump(history, f)

        if (epoch + 1) % 5 == 0:
            # plotter use history values, no need for log
            visual = "history-{:05d}-of-{:05d}.png".format(epoch + 1, num_epochs)
            plot(os.path.join(checkpoint_path, visual), history)

        if epoch > 10:
            if np.mean(history["val loss"][-11:-6]) < np.mean(history["val loss"][-6:]):
                break

    # Save the model
    states = {
        "epoch": epoch + 1,
        "state_dict": net.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(states, os.path.join(checkpoint_path, "final_checkpoint.pth"))

    # eval_loader = get_plain_dataset_loader(
    #     target_size, batch_size, Path(dataset_path).parent / "testing"
    # )

    # eval_hist = validate(eval_loader, num_classes, device, net, criterion)


if __name__ == "__main__":
    config = toml.load("config/train-config.toml")

    num_classes = 2
    lr = config["lr"]
    loss_func = config["loss_func"]
    num_epochs = config["num_epochs"]
    target_size = config["target_size"]
    batch_size = config["batch_size"]

    dataset_path = config["dataset_path"]
    checkpoint_path = config["checkpoint_path"]
    target_type = config["target_type"]
    freeze_pretrained = config["freeze_pretrained"]
    signal_fraction = config["signal_fraction"]
    config["weight"] = [signal_fraction, 1.0]
    weight = [signal_fraction, 1.0]

    # if config["model_path"] != "":
    # model_path = config["model_path"]
    # else:
    # model_path = None

    transform_name = config["transform"]

    Path("results").mkdir(exist_ok=True)

    augs = get_transforms(target_size)
    lr_base = 3e-4
    for lr_factor in (1, 0.1, 0.01):
        lr = lr_base * lr_factor
        config["lr"] = lr
        print("Testing learning rate:", lr)
        for transform_name in augs:
            print("Testing augmentation:", transform_name)
            config["transform"] = transform_name
            # Training a model from scratch
            config["model_path"] = ""
            model_path = ""
            # make dir for checkpoint
            os.makedirs(checkpoint_path, exist_ok=True)
            fname = "results/experiment_" + datetime.datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            config["checkpoint_path"] = fname
            # Write the testing config to file
            with open(checkpoint_path + "/config.toml", "w") as f:
                f.write(toml.dumps(config))

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
            )

            # Move the config and results to a new directory
            shutil.move(checkpoint_path, fname)
            break
