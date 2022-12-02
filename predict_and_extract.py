import argparse
import os

import toml
import torch

from src.extract import intersection
from src.predict import predict

parser = argparse.ArgumentParser()
parser.add_argument(
    "city", help="City to be predicted, must be the same as the name of the dataset"
)
args = parser.parse_args()

config = toml.load("config/best-predict-config.toml")

city_name = args.city
target_type = "green"

tiles_dir = os.path.join("results", "02Images", city_name)
mask_dir = os.path.join("results", "03Masks", target_type, city_name)
tile_size = config["img_size"]

# load checkpoints
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = config["checkpoint_path"]
checkpoint_name = config["checkpoint"]
chkpt = torch.load(os.path.join(checkpoint_path, checkpoint_name), map_location=device)

predict(tiles_dir, mask_dir, tile_size, device, chkpt)

kernel_size_denoise = (
    config["kernel_size_denoise"] if "kernel_size_denoise" in config else 0
)
kernel_size_grow = config["kernel_size_grow"] if "kernel_size_grow" in config else 0

intersection(
    target_type,
    city_name,
    mask_dir,
    kernel_size_denoise=kernel_size_denoise,
    kernel_size_grow=kernel_size_grow,
)
