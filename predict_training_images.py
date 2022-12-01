cd Roofpedia_vc/

# %%
import argparse
import os

import toml
import torch

from src.extract import intersection
from src.predict import predict

config = toml.load("config/predict-config.toml")
config_path = "experiment_20220629_113835/config.toml"
config = toml.load(config_path)
# checkpoint_path = r"C:\Users\ucbqc38\Documents\greenroofs_analysis\data\Roofpedia_220629\model\Green-checkpoint-012-of-100.pth"
# checkpoint_path = "experiment_20220629_113835/Green-checkpoint-002-of-002.pth"
# checkpoint_path = r"C:\Users\ucbqc38\Documents\greenroofs_analysis\data\Roofpedia_220629\Green-checkpoint-021-of-100.pth"
checkpoint_path = r"C:\Users\ucbqc38\Documents\greenroofs_analysis\data\Roofpedia_220629\experiment_20220629_161423\Green-checkpoint-015-of-100.pth"

target_type = "Green"
city_name = "debug"

tiles_dir = os.path.join("dataset", "debug", "images")
tiles_dir = r"C:\Users\ucbqc38\Documents\RoofPedia\data_220529\debug"
# mask_dir = os.path.join("results"   , "debug")
mask_dir = os.path.join("results"   , "debug")
tile_size = config["target_size"]

# load checkpoints
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chkpt = torch.load(checkpoint_path, map_location=device)

predict(tiles_dir, mask_dir, tile_size, device, chkpt)

intersection(target_type, city_name, mask_dir)

# C:\Users\ucbqc38\Documents\RoofPedia\Roofpedia_vc\results\debug\19\261981
