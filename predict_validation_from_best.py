import argparse
import shutil
from pathlib import Path

import toml
import torch
import geopandas as gpd
import pandas as pd

from src.extract import extract
from src.predict import predict

# from imagery_tiling.batched_tiling import tiling_path, native_crs
tiling_path = "./tiling_256_0.25.feather"
native_crs = "EPSG:27700"

parser = argparse.ArgumentParser()
parser.add_argument("config", help="config path")
args = parser.parse_args()

config = toml.load(args.config)

# load checkpoints
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chkpt = torch.load(
    Path(config["checkpoint_path"]) / "final_checkpoint.pth", map_location=device
)

truth_path = Path(
    r"C:\Users\ucbqc38\Documents\RoofPedia\gr_manual_labels_221212.geojson"
)

# for name in ("validation", "training_s", "training_b",):
for name in ("training_b",):

    tiles_dir = Path(config["dataset_path"]) / name
    mask_dir = Path("results") / name / "masks"
    if mask_dir.parent.exists():
        shutil.rmtree(str(mask_dir.parent))
    mask_dir.mkdir(parents=True)
    tile_size = config["target_size"]

    predict(tiles_dir, mask_dir, tile_size, device, chkpt, batch_size=4)

    input_glob = list(mask_dir.glob("*/*png"))
    polygon_output_path = mask_dir.parent / f"{name}.geojson"

    extract(
        input_glob,
        polygon_output_path,
        format="GeoJSON",
    )

    xy = [(float(p.parent.stem), float(p.stem)) for p in input_glob]
    predictions = gpd.read_file(polygon_output_path).set_crs(
        native_crs, allow_override=True
    )  # CRS not set correctly by gdal_polygonize
    gdf_tiles = (
        gpd.read_feather(tiling_path)
        .set_index(["x", "y"])
        .loc[xy][["geometry"]]
        .set_geometry("geometry")
        .to_crs(native_crs)
        .reset_index()
    )
    truth = gpd.read_file(truth_path).to_crs(native_crs)
    print(predictions.crs, truth.crs, gdf_tiles.crs)

    truth = gpd.overlay(truth, gdf_tiles, "intersection")

    if not truth.empty:
        fp = gpd.overlay(predictions, truth, "difference")
    else:
        fp = predictions

    if not fp.empty:
        fp.to_file(polygon_output_path.parent / "fp.geojson", driver="GeoJSON")
    else:
        print("No false positives?")

    if not truth.empty:
        fn = gpd.overlay(truth, predictions, "difference")
        fn.to_file(polygon_output_path.parent / "fn.geojson", driver="GeoJSON")
    else:
        print("No false negatives?")

pd.concat((gpd.read_file(p) for p in Path("results").glob("*/fp.geojson"))).to_file(
    "merged_fp.geojson"
)
pd.concat((gpd.read_file(p) for p in Path("results").glob("*/fn.geojson"))).to_file(
    "merged_fn.geojson"
)
