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
area_limit = 10  # m^2

# load checkpoints
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get building footprints
# I use these for post-processing.
buildings = gpd.read_feather("../../GIS/OS_local_vector/London_buildings.feather")
# But these for counting the buildings.
ukb = gpd.read_feather("../../GIS/ukbuildings_12/ukb_12_geom.feather")

# Get the truth - i.e. the hand produced labels.
# truth_path = Path(
#     r"C:\Users\ucbqc38\Documents\RoofPedia\gr_manual_labels_230104.geojson"
# )
# truth = gpd.read_file(truth_path).to_crs(native_crs)
# truth = gpd.GeoDataFrame(geometry=truth.geometry.explode(index_parts=False), crs=truth.crs) # Fix self intersection which is my fault.
# truth = gpd.overlay(truth, buildings)
# truth.to_file("truth_exploded.geojson")

truth_2021 = gpd.read_file("../labels_rename/gr_manual_labels_2021.geojson").to_crs(
    native_crs
)
truth_2021 = gpd.GeoDataFrame(
    geometry=truth_2021.geometry.explode(index_parts=False), crs=truth_2021.crs
)  # Fix self intersection which is my fault.
truth_2019 = gpd.read_file("../labels_rename/gr_manual_labels_2019.gpkg").to_crs(
    native_crs
)
truth_2019 = gpd.GeoDataFrame(
    geometry=truth_2019.geometry.explode(index_parts=False), crs=truth_2019.crs
)  # Fix self intersection which is my fault.

labelling_area = gpd.read_file("../labels_rename/selected_area_220404.geojson").to_crs(
    native_crs
)

# %%
results = []
# Iterate over kfolds.
# Load config
config = "alt_results/alt_experiment_20230125_143332/config.toml"
print(config)
config = toml.load(config)

# Data which will be used to test it.
dataset_path = Path("dataset")

# Load model
chkpt = torch.load(
    Path(config["checkpoint_path"]) / "final_checkpoint.pth", map_location=device
)


# Iterate over datasets.
for name in ("testing", "testing_alt"):
    print(f"dataset={name}")

    if name == "testing":
        ds_dir = Path(dataset_path) / "testing"
    elif name == "testing_alt":
        ds_dir = Path("alt_dataset") / "testing"
    else:
        raise ValueError()

    if "alt" in name:
        print(f"{name} uses 2019 labels")
        truth = truth_2019
    else:
        truth = truth_2021
        print(f"{name} uses 2021 labels")

    # Set up directories
    tiles_dir = ds_dir
    mask_dir = Path("alt_results") / f"k{config['kfold']}" / name / "masks"
    polygon_output_path = mask_dir.parent / f"{name}.geojson"

    if mask_dir.parent.exists():
        shutil.rmtree(str(mask_dir.parent))
    mask_dir.mkdir(parents=True)
    tile_size = config["target_size"]

    print("Prediction")
    print(tiles_dir, mask_dir)

    predict(tiles_dir, mask_dir, tile_size, device, chkpt, batch_size=4)
    input_glob = list(mask_dir.glob("*/*png"))

    print("Extraction")
    extract(
        input_glob,
        polygon_output_path,
    )

    xy = [(float(p.parent.stem), float(p.stem)) for p in input_glob]
    predictions = gpd.read_file(polygon_output_path).set_crs(
        native_crs, allow_override=True
    )

    # Join across 0 width gaps
    predictions = gpd.GeoDataFrame(
        geometry=list(predictions.buffer(0).unary_union.geoms), crs=predictions.crs
    )

    # Drop tiny polygons
    predictions = predictions[predictions.area > area_limit]

    gdf_tiles = (
        gpd.read_feather(tiling_path)
        .pipe(lambda _gdf: _gdf[_gdf.within(labelling_area.unary_union)])
        .set_index(["x", "y"])
        .loc[xy][["geometry"]]
        .set_geometry("geometry")
        .to_crs(native_crs)
        .reset_index()
    )
    gdf_tiles_signal = gdf_tiles.sjoin(
        predictions[["geometry"]].assign(pred=True)
    ).pipe(lambda _df: _df[_df.pred == True])[["geometry"]]
    total_area = gdf_tiles.area.sum()
    total_buildings_count = gdf_tiles.sjoin(ukb).geomni_premise_id.unique().shape[0]

    # Remove predictions outside the selected area.
    predictions = gpd.overlay(predictions, gdf_tiles)
    # Remove predictions outside building footprints.
    predictions_bui = gpd.overlay(predictions, buildings)
    # Remove truth outside the selected area.
    truth_local = gpd.overlay(truth, gdf_tiles, "intersection", keep_geom_type=True)
    truth_local = gpd.overlay(
        truth_local, buildings, "intersection", keep_geom_type=True
    )

    positive_predictions = predictions.area.sum()
    local_buildings = gpd.overlay(
        gdf_tiles, buildings, "intersection", keep_geom_type=True
    )
    total_buildings_area = local_buildings.area.sum()

    print("Truth overlay")
    if name != "training_b":
        # If the building intersection is used.
        fp_bui = gpd.overlay(predictions_bui, truth_local, "difference")
        tp_bui = gpd.overlay(
            predictions_bui, truth_local, "intersection", keep_geom_type=True
        )
        fn_bui = gpd.overlay(truth_local, predictions_bui, "difference")
        union_bui = gpd.overlay(predictions_bui, truth_local, "union")
        tn_bui = gpd.overlay(gdf_tiles, union_bui, "difference")

        # If it isn't used.
        fp = gpd.overlay(predictions, truth_local, "difference")
        tp = gpd.overlay(predictions, truth_local, "intersection", keep_geom_type=True)
        fn = gpd.overlay(truth_local, predictions, "difference")
        union = gpd.overlay(predictions, truth_local, "union")
        tn = gpd.overlay(gdf_tiles, union, "difference")

        # Count area.
        fp_area = fp.area.sum()
        tp_area = tp.area.sum()
        fn_area = fn.area.sum()
        tn_area = tn.area.sum()
        union_area = union.area.sum()

        fp_bui_area = fp_bui.area.sum()
        tp_bui_area = tp_bui.area.sum()
        fn_bui_area = fn_bui.area.sum()
        tn_bui_area = tn_bui.area.sum()
        union_bui_area = union_bui.area.sum()

        # Count buildings.
        truth_buildings = truth_local.sjoin(ukb).geomni_premise_id.unique()
        pred_buildings = predictions_bui.sjoin(ukb).geomni_premise_id.unique()
        tp_count = np.intersect1d(truth_buildings, pred_buildings).shape[0]
        fp_count = np.setdiff1d(pred_buildings, truth_buildings).shape[0]
        fn_count = np.setdiff1d(truth_buildings, pred_buildings).shape[0]
        union_buildings = np.union1d(truth_buildings, pred_buildings)
        tn_count = np.setdiff1d(
            gdf_tiles.sjoin(ukb).geomni_premise_id.unique(), union_buildings
        ).shape[0]
        union_count = union_buildings.shape[0]

        # If only positive tiles are included.
        truth_pos = gpd.overlay(
            truth_local, gdf_tiles_signal, "intersection", keep_geom_type=True
        )
        predictions_bui_pos = gpd.overlay(
            predictions_bui, gdf_tiles_signal, "intersection", keep_geom_type=True
        )
        truth_buildings = truth_pos.sjoin(ukb).geomni_premise_id.unique()
        pred_buildings = predictions_bui_pos.sjoin(ukb).geomni_premise_id.unique()
        tp_pos_count = np.intersect1d(truth_buildings, pred_buildings).shape[0]
        fp_pos_count = np.setdiff1d(pred_buildings, truth_buildings).shape[0]
        fn_pos_count = np.setdiff1d(truth_buildings, pred_buildings).shape[0]
        union_buildings = np.union1d(truth_buildings, pred_buildings)
        tn_pos_count = np.setdiff1d(
            gdf_tiles.sjoin(ukb).geomni_premise_id.unique(), union_buildings
        ).shape[0]
        union_pos_count = union_buildings.shape[0]

    else:
        print("No truth geometry")
        fp = predictions
        fp_bui = predictions_bui
        fp_area, fp_bui_area = fp.area.sum(), fp_bui.area.sum()
        fn_area, fn_bui_area = 0, 0
        tp_area, tp_bui_area = 0, 0
        fp_count = fp.sjoin(ukb).geomni_premise_id.unique().shape[0]
        tp_count = 0
        fn_count = 0
        union_area, union_bui_area = predictions.area.sum(), predictions_bui.area.sum()
        fp_pos_count = (
            gpd.overlay(fp_bui, gdf_tiles_signal, "intersection", keep_geom_type=True)
            .sjoin(ukb)
            .geomni_premise_id.unique()
            .shape[0]
        )

    results.append(
        {
            "tp_area": tp_area,
            "fp_area": fp_area,
            "fn_area": fn_area,
            "tn_area": tn_area,
            "tp_bui_area": tp_bui_area,
            "fp_bui_area": fp_bui_area,
            "fn_bui_area": fn_bui_area,
            "tn_bui_area": tn_bui_area,
            "fp_count": fp_count,
            "tp_count": tp_count,
            "fn_count": fn_count,
            "tn_count": tn_count,
            "fp_pos_count": fp_pos_count,
            "tp_pos_count": tp_pos_count,
            "fn_pos_count": fn_pos_count,
            "tn_pos_count": tn_pos_count,
            "total_area": total_area,
            "total_buildings_count": total_buildings_count,
            "built_area": total_buildings_area,
            "union_area": union_area,
            "union_count": union_count,
            "union_pos_count": union_pos_count,
            "positive_predict_area": positive_predictions,
            "kfold": config["kfold"],
            "ds": name,
        }
    )

    df = pd.DataFrame(results)
    df.to_csv("alt_results/alt_vector_confusion.csv")

    epsilon = 1e-10
    df["precision_bui_area"] = df.tp_bui_area / (
        df.tp_bui_area + df.fp_bui_area + epsilon
    )
    df["recall_bui_area"] = df.tp_bui_area / (df.tp_bui_area + df.fn_bui_area + epsilon)
    df["f1_bui_area"] = (
        2
        * (df.precision_bui_area * df.recall_bui_area)
        / (df.precision_bui_area + df.recall_bui_area + epsilon)
    )
    print(
        df[
            ["kfold", "ds", "precision_bui_area", "recall_bui_area", "f1_bui_area"]
        ].to_string()
    )

# %%
import pandas as pd

# %%
df = pd.read_csv("alt_results/alt_vector_confusion.csv")
df = df.set_index(["kfold", "ds"]).sort_index()
df = df.reset_index()

for k in range(1, 5):
    _df = pd.DataFrame(
        df[(df.kfold == k) & (df.ds == "training_b")].values
        + df[(df.kfold == k) & (df.ds == "training_s")].values,
        columns=df.columns,
    )
    _df["kfold"] = k
    _df["ds"] = "training"
    df = pd.concat((df, _df))


# Make an average across k folds
for ds in df.ds.unique():
    _df = pd.DataFrame(df[df.ds == ds].mean()).T.assign(kfold="average", ds=ds)
    df = pd.concat((df, _df))

df = df.set_index(["ds", "kfold"]).sort_index()

epsilon = 1e-10  # prevent zero division error for training_b
df["precision_bui_area"] = df.tp_bui_area / (df.tp_bui_area + df.fp_bui_area + epsilon)
df["recall_bui_area"] = df.tp_bui_area / (df.tp_bui_area + df.fn_bui_area + epsilon)
df["f1_bui_area"] = (
    2
    * (df.precision_bui_area * df.recall_bui_area)
    / (df.precision_bui_area + df.recall_bui_area + epsilon)
)
df["precision_area"] = df.tp_area / (df.tp_area + df.fp_area + epsilon)
df["recall_area"] = df.tp_area / (df.tp_area + df.fn_area + epsilon)
df["f1_area"] = (
    2
    * (df.precision_area * df.recall_area)
    / (df.precision_area + df.recall_area + epsilon)
)
df["precision_count"] = df.tp_count / (df.tp_count + df.fp_count + epsilon)
df["recall_count"] = df.tp_count / (df.tp_count + df.fn_count + epsilon)
df["f1_count"] = (
    2
    * (df.precision_count * df.recall_count)
    / (df.precision_count + df.recall_count + epsilon)
)
df["precision_pos_count"] = df.tp_pos_count / (
    df.tp_pos_count + df.fp_pos_count + epsilon
)
df["recall_pos_count"] = df.tp_pos_count / (df.tp_pos_count + df.fn_pos_count + epsilon)
df["f1_pos_count"] = (
    2
    * (df.precision_pos_count * df.recall_pos_count)
    / (df.precision_pos_count + df.recall_pos_count + epsilon)
)

# %%
df = df.assign(
    fp=df.fp_bui_area / df.built_area,
    tp=df.tp_bui_area / df.built_area,
    fn=df.fn_bui_area / df.built_area,
    tn=df.tn_bui_area / df.built_area,
)
df = df.assign(tn=1 - df.fp - df.fn - df.tp)
df = df.assign(
    iou_bui_area=(df.tp / (df.tp + df.fp + df.fn)),
    accuracy_bui_area=(1 - df.fn - df.fp),
)

with open("alt_results/accuracy_area_long.tex", "w") as w:
    w.write(
        df.reset_index()
        .set_index(["ds", "kfold"])[
            [
                "accuracy_bui_area",
                "iou_bui_area",
                "precision_bui_area",
                "recall_bui_area",
                "f1_bui_area",
            ]
        ]
        .round(4)
        .to_latex()
    )
# %%
