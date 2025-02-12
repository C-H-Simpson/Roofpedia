import os
import torch
import toml
import argparse

from src.predict import predict
from src.extract import intersection

parser = argparse.ArgumentParser()
parser.add_argument("city", help="City to be predicted, must be the same as the name of the dataset")
args = parser.parse_args()

config = toml.load('config/predict-config.toml')
config_path = "experiment_20220407_151148/config.toml"
config= toml.load(config_path)
checkpoint_path = "experiment_20220407_151148/Green-checkpoint-100-of-100.pth"
    
city_name = args.city
target_type = "Green"

tiles_dir = os.path.join("results", '02Images', city_name)
mask_dir = os.path.join("results", "03Masks", target_type, city_name)
tile_size =  config["target_size"]

# load checkpoints
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chkpt = torch.load(checkpoint_path, map_location=device)

predict(tiles_dir, mask_dir, tile_size, device, chkpt, batch_size=64)

intersection(target_type, city_name, mask_dir, kernel_size_denoise=0, kernel_size_grow=0)
