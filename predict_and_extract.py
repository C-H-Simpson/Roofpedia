import argparse
import os
from pathlib import Path

import toml
import torch

from src.extract import extract
from src.predict import predict

parser = argparse.ArgumentParser()
parser.add_argument(
    "city", help="City to be predicted, must be the same as the name of the dataset"
)
args = parser.parse_args()

config = toml.load("config/best-predict-config.toml")

city = args.city
target_type = "green"

tiles_dir = os.path.join("results", "02Images", city)
mask_dir = os.path.join("results", "03Masks", target_type, city)
tile_size = config["img_size"]

# load checkpoints
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = config["checkpoint_path"]
checkpoint_name = config["checkpoint"]
chkpt = torch.load(os.path.join(checkpoint_path, checkpoint_name), map_location=device)

predict(tiles_dir, mask_dir, tile_size, device, chkpt)

format = "GeoJSON"
polygon_output_path = Path("results") / args.city / "polygons.geojson"
merged_raster_path = Path("results") / args.city / "merged.tif"

mask_glob = list((Path("results") / args.city / "predictions").glob("*/*png"))

extract(
    mask_glob=mask_glob,
    polygon_output_path=polygon_output_path,
    merged_raster_path=merged_raster_path,
    format=format,
)
