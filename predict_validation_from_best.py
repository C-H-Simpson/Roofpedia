import argparse
import os

import toml
import torch

from src.extract import intersection
from src.predict import predict

parser = argparse.ArgumentParser()
parser.add_argument("config", help="config path")
parser.add_argument("checkpoint", help="checkpoint path")
args = parser.parse_args()

# config_path = "experiment_20220407_151148/config.toml"
config = toml.load(args.config)
# checkpoint_path = "experiment_20220407_151148/Green-checkpoint-100-of-100.pth"

target_type = "Green"
city_name = "validation"

tiles_dir = os.path.join(config["dataset_path"], "validation", "images")
mask_dir = os.path.join("results", "validation")
tile_size = config["target_size"]

# load checkpoints
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chkpt = torch.load(args.checkpoint, map_location=device)

predict(tiles_dir, mask_dir, tile_size, device, chkpt)

intersection(target_type, city_name, mask_dir)
