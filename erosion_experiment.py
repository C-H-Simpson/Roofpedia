# %%
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import toml
import geopandas as gpd
import pandas as pd

from src.extract import extract

gpd.options.use_pygeos = True

# %%
tiling_path = "./tiling_256_0.25.feather"
native_crs = "EPSG:27700"

# load checkpoints
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the base config
config = toml.load("config/best-predict-config.toml")

# Get building footprints
# I use these for post-processing.
buildings = gpd.read_feather("../../GIS/OS_local_vector/London_buildings.feather")
# But these for counting the buildings.
ukb = gpd.read_feather("../../GIS/ukbuildings_12/ukb_12_geom.feather")

# Get the truth - i.e. the hand produced labels.
truth_path = Path("../labels_rename/gr_manual_labels_2021.geojson")
truth = gpd.read_file(truth_path).to_crs(native_crs)

# %%
results = []
config = "config/best-predict-config.toml"

# Load config
print(config)
config = toml.load(config)

# Load model
# chkpt = torch.load(
#     Path(config["checkpoint_path"]) / "final_checkpoint.pth", map_location=device
# )

name = "validation"

# %%
# for simplify in (0, 0.25, 0.5, 1):
erosion = 0
simplify = 0.25
truth_expansion = 0
opening = 0
closing = 0
for area_limit in (0, 1, 5, 10, 20, 50, 100):
    print(f"{opening=} {closing=}")
    # for erosion in (-2, -1,  0):
    # for truth_expansion in (0, 1, 2):
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
    polygon_output_path = Path("erosion_experiment") / f"{name}.geojson"
    polygon_output_path.parent.mkdir(exist_ok=True, parents=False)
    input_glob = list(mask_dir.glob("*/*png"))

    print("Extraction")
    extract(
        input_glob,
        polygon_output_path,
        opening=opening,
        closing=closing,
    )

    xy = [(float(p.parent.stem), float(p.stem)) for p in input_glob]
    predictions = gpd.read_file(polygon_output_path).set_crs(
        native_crs, allow_override=True
    )  # CRS not set correctly by gdal_polygonize

    # Apply operations
    predictions = gpd.GeoDataFrame(
        geometry=list(predictions.unary_union.geoms), crs=native_crs
    )
    if simplify != 0:
        predictions = gpd.GeoDataFrame(geometry=predictions.geometry.simplify(simplify))
    if erosion != 0:
        predictions = gpd.GeoDataFrame(
            geometry=list(predictions.unary_union.buffer(-erosion).geoms),
            crs=native_crs,
        )

    predictions = predictions[predictions.area > area_limit]

    gdf_tiles = (
        gpd.read_feather(tiling_path)
        .set_index(["x", "y"])
        .loc[xy][["geometry"]]
        .set_geometry("geometry")
        .to_crs(native_crs)
        .reset_index()
    )
    total_area = gdf_tiles.area.sum()
    total_buildings_count = gdf_tiles.sjoin(ukb).geomni_premise_id.unique().shape[0]

    # Remove predictions outside the selected area.
    predictions = gpd.overlay(predictions, gdf_tiles)
    # Remove predictions outside building footprints.
    predictions = gpd.overlay(predictions, buildings)
    # Remove truth outside the selected area.
    truth_local = gpd.overlay(truth, gdf_tiles, "intersection")
    truth_local = truth_local.assign(geometry=truth_local.buffer(truth_expansion))
    truth_local = gpd.overlay(truth_local, buildings, "intersection")

    positive_predictions = predictions.area.sum()
    local_buildings = gpd.overlay(gdf_tiles, buildings, "intersection")
    total_buildings_area = local_buildings.area.sum()

    if name != "training_b":
        fp = gpd.overlay(predictions, truth_local, "difference")
        tp = gpd.overlay(predictions, truth_local, "intersection")
        fn = gpd.overlay(truth_local, predictions, "difference")
        union = gpd.overlay(predictions, truth_local, "union")
        tn = gpd.overlay(gdf_tiles, union, "difference")

        # Count area.
        fp_area = fp.area.sum()
        tp_area = tp.area.sum()
        fn_area = fn.area.sum()
        tn_area = tn.area.sum()
        union_area = union.area.sum()

        # Count buildings.
        truth_buildings = truth_local.sjoin(ukb).geomni_premise_id.unique()
        pred_buildings = predictions.sjoin(ukb).geomni_premise_id.unique()
        tp_count = np.intersect1d(truth_buildings, pred_buildings).shape[0]
        fp_count = np.setdiff1d(pred_buildings, truth_buildings).shape[0]
        fn_count = np.setdiff1d(truth_buildings, pred_buildings).shape[0]
        union_buildings = np.union1d(truth_buildings, pred_buildings)
        tn_count = np.setdiff1d(
            gdf_tiles.sjoin(ukb).geomni_premise_id.unique(), union_buildings
        ).shape[0]
        union_count = union_buildings.shape[0]

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
        union_area = predictions.area.sum()

    results.append(
        {
            "tp_area": tp_area,
            "fp_area": fp_area,
            "fn_area": fn_area,
            "tn_area": tn_area,
            "fp_count": fp_count,
            "tp_count": tp_count,
            "fn_count": fn_count,
            "tn_count": tn_count,
            "total_area": total_area,
            "total_buildings_count": total_buildings_count,
            "built_area": total_buildings_area,
            "union_area": union_area,
            "union_count": union_count,
            "positive_predict_area": positive_predictions,
            "kfold": config["kfold"],
            "ds": name,
            "precision": tp_area / (tp_area + fp_area),
            "recall": tp_area / (tp_area + fn_area),
            "erosion": erosion,
            "truth_expansion": truth_expansion,
            "simplify": simplify,
            "opening": opening,
            "closing": closing,
            "area_limit": area_limit,
        }
    )
    df = pd.DataFrame(results)
    # df.to_csv("erosion_experiment.csv")
    epsilon = 1e-10  # prevent zero division error for training_b
    df["f1"] = 2 * (df.precision * df.recall) / (df.precision + df.recall + epsilon)
    print(df[["opening", "closing", "precision", "recall", "f1"]])
    # %%
    print(df[["opening", "closing", "precision", "recall", "f1"]])


# %%
print(df[["erosion", "simplify", "truth_expansion", "fp_area", "fn_area"]])
# %%
import pandas as pd

df = pd.read_csv("erosion_experiment.csv")
# %%
epsilon = 1e-10  # prevent zero division error for training_b
df["f1"] = 2 * (df.precision * df.recall) / (df.precision + df.recall + epsilon)

# %%
print(
    df[["erosion", "simplify", "precision", "recall", "f1"]]
    .sort_values("f1")
    .to_string()
)
# %%
fig, ax = plt.subplots()
df.plot.scatter("erosion", "f1", ax=ax)
ax.set_xlabel("Erosion distance (m)")
ax.set_ylabel("F-score")
fig.savefig("erosion.png", dpi=300)
fig.savefig("erosion.pdf")

# %%
