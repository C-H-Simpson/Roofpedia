import argparse
import shutil
import tempfile
from pathlib import Path

import toml
import torch

from src.extract import extract
from src.predict import predict

native_crs = "EPSG:27700"

parser = argparse.ArgumentParser()
parser.add_argument(
    "gref", help="Grid reference to be extracted"
)
parser.add_argument("config", help="config path", default="config/best-predict-config.toml")
args = parser.parse_args()

config = toml.load(args.config)

# This will work with either CPU or GPU, allowing for parallelisation.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = config["batch_size"] if torch.cuda.is_available() else 1

# load checkpoints
chkpt = torch.load(
    Path(config["checkpoint_path"]) / "final_checkpoint.pth", map_location=device
)

tiles_parent_dir = Path("/home/ucbqc38/Scratch")
for name in ("getmapping_2019_tiled", "getmapping_2021_tiled"):
    ds_path = tiles_parent_dir / name

    tiles_dir = tiles_parent_dir / name
    mask_dir = tiles_parent_dir / "results" / name / "masks"
    if mask_dir.parent.exists():
        shutil.rmtree(str(mask_dir.parent))
    mask_dir.mkdir(parents=True)

    tile_size = config["target_size"]

    predict(tiles_dir, mask_dir, tile_size, device, chkpt, batch_size=batch_size)

    input_glob = list(mask_dir.glob("*/*png"))
    polygon_output_path = mask_dir.parent / f"{name}.geojson"

    extract(
        input_glob,
        polygon_output_path,
        format="GeoJSON",
    )