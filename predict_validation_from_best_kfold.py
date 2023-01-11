# %%
import json
import numpy as np
from pathlib import Path
import shutil

import toml
import torch
import geopandas as gpd
import pandas as pd

from src.extract import extract
from src.predict import predict

gpd.options.use_pygeos = True

# %%
tiling_path = "./tiling_256_0.25.feather"
native_crs = "EPSG:27700"

# load checkpoints
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the base config
config = toml.load("config/best-predict-config.toml")

# Get building footprints
# I use these for post-processing.
buildings = gpd.read_feather("../../GIS/OS_local_vector/London_buildings.feather")
# But these for counting the buildings.
ukb = gpd.read_feather("../../GIS/ukbuildings_12/ukb_12_geom.feather")

# Get the truth - i.e. the hand produced labels.
truth_path = Path(
    r"C:\Users\ucbqc38\Documents\RoofPedia\gr_manual_labels_230104.geojson"
)
truth = gpd.read_file(truth_path).to_crs(native_crs)
truth = gpd.GeoDataFrame(geometry=truth.geometry.explode(index_parts=False), crs=truth.crs) # Fix self intersection which is my fault.
truth = gpd.overlay(truth, buildings)
truth.to_file("truth_exploded.geojson")


# We will iterate over the "best config" and its kfolds
original_config = toml.load("config/best-predict-config.toml")
kfold_config_paths = list(Path("results").glob("kfold_*/config.toml"))
assert kfold_config_paths, "did kfold already get run?"
print(f"{len(kfold_config_paths)=}")

# Check the kfold files match the original config.
for p in kfold_config_paths:
    print(p)
    config = toml.load(p)
    for key in original_config:
        if key in ("dataset_path", "checkpoint_path", "kfold", "alt_validation_path", "weight"):
            continue
        if config[key] != original_config[key]:
            raise ValueError(
                "Non matching kfold config"
                + f"{p} {key} {config[key]} != {original_config[key]}"
            )

# %%
results = []
# Iterate over kfolds.
for config in kfold_config_paths + ["config/best-predict-config.toml"]:
    # Load config
    print(config)
    config = toml.load(config)

    # Load model
    chkpt = torch.load(
        Path(config["checkpoint_path"]) / "final_checkpoint.pth", map_location=device
    )

    # Iterate over datasets.
    for name in ("validation", "training_s", "training_b", "testing", "testing_alt", "validation_alt"):
        print(f"dataset={name}")

        if name == "testing":
            ds_dir = Path(config["dataset_path"]).parent / "testing"
        elif name == "testing_alt":
            ds_dir = Path(config["dataset_path"]).parent / "testing_alt"
        else:
            ds_dir = Path(config["dataset_path"]) / name

        # Set up directories
        tiles_dir = ds_dir
        mask_dir = Path("results") / f"k{config['kfold']}" / name / "masks"
        polygon_output_path = mask_dir.parent / f"{name}.geojson"

        if mask_dir.parent.exists():
            shutil.rmtree(str(mask_dir.parent))
        mask_dir.mkdir(parents=True)
        tile_size = config["target_size"]

        print(tiles_dir, mask_dir)
        predict(tiles_dir, mask_dir, tile_size, device, chkpt, batch_size=4)

        input_glob = list(mask_dir.glob("*/*png"))

        print("Extraction")
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
        total_area = gdf_tiles.area.sum()
        total_buildings = gdf_tiles.sjoin(ukb).geomni_premise_id.unique().shape[0]

        # Remove predictions outside the selected area.
        predictions = gpd.overlay(predictions, gdf_tiles)
        # Remove predictions outside building footprints.
        predictions = gpd.overlay(predictions, buildings)
        # Remove truth outside the selected area.
        truth_local = gpd.overlay(truth, gdf_tiles, "intersection")

        positive_predictions = predictions.area.sum()
        total_buildings_area = gpd.overlay(gdf_tiles, buildings, "intersection").area.sum()

        print("Truth overlay")
        if name != "training_b":
            truth_local.to_file(polygon_output_path.parent / "truth_local.geojson", driver="GeoJSON")
            print("gen fp")
            fp = gpd.overlay(predictions, truth_local, "difference")
            tp = gpd.overlay(predictions, truth_local, "intersection")
            print("gen fn")
            fn = gpd.overlay(truth_local, predictions, "difference")
            if not fn.empty:
                fn.to_file(polygon_output_path.parent / "fn.geojson", driver="GeoJSON")
            else:
                print("No false negatives?")

            # Count area.
            # This seems to lead to results where tp+fp>1
            fp_area = fp.area.sum()
            tp_area = tp.area.sum()
            fn_area = fn.area.sum()
            union = gpd.overlay(predictions, truth_local, "union").area.sum()

            # Count buildings.
            truth_buildings = truth_local.sjoin(ukb).geomni_premise_id.unique()
            pred_buildings = predictions.sjoin(ukb).geomni_premise_id.unique()
            tp_count = np.intersect1d(truth_buildings, pred_buildings).shape[0]
            fp_count = np.setdiff1d(pred_buildings, truth_buildings).shape[0]
            fn_count = np.setdiff1d(truth_buildings, pred_buildings).shape[0]
            union_count = np.union1d(truth_buildings, pred_buildings).shape[0]

        else:
            print("No truth geometry")
            fp = predictions
            fn = None
            tp = None
            fp_area = fp.area.sum()
            fn_area = 0
            tp_area = 0
            fp_count = fp.sjoin(ukb).geomni_premise_id.unique().shape[0]
            tp_count = 0
            fn_count = 0
            union = predictions.area.sum()

        if not fp.empty:
            fp.to_file(polygon_output_path.parent / "fp.geojson", driver="GeoJSON")
        else:
            print("No false positives?")

        if not tp is None:
            if not tp.empty:
                tp.to_file(polygon_output_path.parent / "tp.geojson")

        results.append ({
            "tp_area": tp_area, "fp_area": fp_area, "fn_area": fn_area, "total_area": total_area,
            "fp_count": fp_count, "tp_count": tp_count, "fn_count": fn_count, "total_buildings": total_buildings,
            "union": union, "union_count": union_count,
            "positive_predict_area": positive_predictions,
            "kfold": config["kfold"], "ds": name
        })

        pd.DataFrame(results).to_csv("kfold_vector_confusion.csv")



# %%
import pandas as pd
import json

# %%
df = pd.read_csv("kfold_vector_confusion.csv")
df

# # %%
# with open("kfold_vector_confusion.json", 'r') as f:
#     data = json.load(f)
# agg = []
# for key, val in data.items():
#     for d, v in val.items():
#         agg.append(pd.DataFrame({**v, "kfold": key, "ds": d}))

# df = pd.concat(agg)
# df.to_csv("kfold_vector_confusion.csv")
# df

# # %%
# df = pd.read_csv("kfold_vector_confusion_manualtranspose.csv")
# df = df.rename(columns={"Unnamed: 0": "ds"})
df = df.set_index(["kfold", "ds"]).sort_index()
df = df.reset_index()

# %%
for k in range(1,5):
    _df = pd.DataFrame(
        df[(df.kfold==k) & (df.ds=="training_b")].values + df[(df.kfold==k) & (df.ds=="training_s")].values,
        columns=df.columns
    )
    _df["kfold"] = k
    _df["ds"] = "training"
    df = pd.concat((df, _df))

# %%
df = df[df.ds!="training_b"]

# %%
df = df.set_index(["kfold", "ds"]).sort_index()


# %%
df["precision_area"] = df.tp_area / (df.tp_area + df.fp_area)
df["precision_count"] = df.tp_count / (df.tp_count + df.fp_count)
df["recall_area"] = df.tp_area / (df.tp_area + df.fn_area)
df["recall_count"] = df.tp_count / (df.tp_count + df.fn_count)
df["f1_area"] = 2*(df.precision_area*df.recall_area)/(df.precision_area+df.recall_area)
df["f1_count"] = 2*(df.precision_count*df.recall_count)/(df.precision_count+df.recall_count)

# %%
df.to_csv("kfold_vector_confusion_format.csv")

# %%
## Format the tables for the paper ##

# %%
import pandas as pd
df = pd.read_csv("kfold_vector_confusion_format.csv")
print(df)
# %%
# Get the built area, which we forgot to save.
import geopandas as gpd
import toml
from pathlib import Path
original_config = toml.load("config/best-predict-config.toml")
kfold_config_paths = list(Path("results").glob("kfold_*/config.toml"))
tiling_path = "./tiling_256_0.25.feather"
native_crs = "EPSG:27700"
buildings = gpd.read_feather("../../GIS/OS_local_vector/London_buildings.feather")
building_areas = []
for config in kfold_config_paths + ["config/best-predict-config.toml"]:
    # Load config
    print(config)
    config = toml.load(config)

    # Iterate over datasets.
    for name in ("validation", "training_s", "training_b", "testing", "testing_alt"):
        print(f"dataset={name}")

        mask_dir = Path("results") / f"k{config['kfold']}" / name / "masks"
        input_glob = list(mask_dir.glob("*/*png"))

        xy = [(float(p.parent.stem), float(p.stem)) for p in input_glob]
        gdf_tiles = (
            gpd.read_feather(tiling_path)
            .set_index(["x", "y"])
            .loc[xy][["geometry"]]
            .set_geometry("geometry")
            .to_crs(native_crs)
            .reset_index()
        )
        total_buildings_area = gpd.overlay(gdf_tiles, buildings, "intersection").area.sum()
        building_areas.append({"kfold": config["kfold"], "ds": name, "built_area": total_buildings_area})
# %%
building_areas = pd.DataFrame(building_areas)
# %%
building_areas = pd.concat((
    building_areas.set_index("kfold"),
    (building_areas[building_areas.ds=="training_b"].set_index("kfold") + building_areas[building_areas.ds=="training_s"].set_index("kfold")).assign(ds="training")
)).reset_index()
# %%
# df = df.drop(columns=["built_area"])

# %%
df = df.reset_index().set_index(["kfold", "ds"]).join(building_areas.set_index(["kfold", "ds"]))
# %%
df = df.assign(tn_area=df.built_area - df.tp_area - df.fn_area - df.fp_area)
# %%
df = df.assign(fp=df.fp_area/df.built_area, tp=df.tp_area/df.built_area, fn=df.fn_area/df.built_area, tn=df.tn_area/df.built_area)
# %%
df[["total_area", "built_area", "tp", "tn", "fp", "fn"]]
# %%
print(
    df[["total_area", "built_area", "tp", "tn", "fp", "fn"]].to_latex(
        formatters={
            "total_area": lambda _f: f"{_f/1e6:0.1f}",
            "built_area": lambda _f: f"{_f/1e6:0.3f}",
            "tp": lambda _f: f"{_f:0.4f}",
            "tn": lambda _f: f"{_f:0.4f}",
            "fp": lambda _f: f"{_f:0.4f}",
            "fn": lambda _f: f"{_f:0.4f}",
        }
    )
)
# %%
df.columns
# %%
df = df.assign(iou_area=(df.tp_area/df.union), accuracy_area=(df.tp+df.tn))
# %%
print(
    df[["accuracy_area", "iou_area", "precision_area", "recall_area", "f1_area"]]
    .round(4)
    .to_latex()
)


# %%
fp = gpd.read_file(r"C:\Users\ucbqc38\Documents\RoofPedia\Roofpedia_vc\results\k1\training_s\fp.geojson")
fn = gpd.read_file(r"C:\Users\ucbqc38\Documents\RoofPedia\Roofpedia_vc\results\k1\training_s\fn.geojson")
tp = gpd.read_file(r"C:\Users\ucbqc38\Documents\RoofPedia\Roofpedia_vc\results\k1\training_s\tp.geojson")

# %%
fp.area.sum()
# %%
fn.area.sum()
# %%
tp.area.sum()

# %%
df[["tp_area", "fp_area", "fn_area"]].round(1)