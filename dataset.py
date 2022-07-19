import argparse
import glob
import os
import random
import shutil
from pathlib import Path

import cv2
import geopandas as gpd
import joblib
import mercantile
import numpy as np
import shapely
from PIL import Image
from tqdm import tqdm

from src.colors import make_palette

random_obj = random.Random(144)
# %%


def load_img(label_path, source_path):
    print(label_path, source_path)
    signal_labels = [str(p) for p in Path(label_path).glob("*/*/*.png")]
    signal_images = [str(p) for p in Path(source_path).glob("*/*/*.png")]
    print(str(len(signal_labels)) + " label files found")
    print(str(len(signal_images)) + " source files found")
    return signal_labels, signal_images


def select_tiles(
    training_area_path, keep_signal_proportion=1, keep_background_proportion=0
):
    # Load the limits of the labelled area.
    gdf_labelled_area = gpd.read_file(training_area_path).to_crs("EPSG:4326")

    # Load a glob of all the label tiles, some of which won't be valid.
    label_tiles = list(Path("dataset/labels").glob("*/*/*png"))

    # Check the training area returned is correct
    tiles = [
        (int(Path(s).parent.stem), int(Path(s).stem), int(Path(s).parent.parent.stem))
        for s in label_tiles
    ]
    bounds = [mercantile.bounds(*s) for s in tiles]
    boxes = [shapely.geometry.box(*b) for b in bounds]
    gdf = gpd.GeoDataFrame(
        {"label_tiles": [str(a) for a in label_tiles]}, geometry=boxes, crs="EPSG:4326"
    )
    # gdf.to_file("dataset/label_tiles_return.geojson", driver="GeoJSON")

    # Limit to the valid labelled area.
    intersect = gpd.overlay(gdf, gdf_labelled_area)
    intersect.to_file("dataset/labelled_tiles_return.geojson", driver="GeoJSON")
    del gdf

    label_tiles = np.unique(intersect.label_tiles.values)

    print(f"{len(label_tiles)} tiles in training area")

    # Find non-background tiles within the training area.
    any_list = [cv2.imread(p).any() for p in tqdm(label_tiles, desc="backgrounds")]
    print(f"Labels exist for {sum(any_list)} tiles")
    if not any_list:
        raise ValueError("All tiles are background")
    background_label = [p for p, i in zip(label_tiles, any_list) if not i]
    signal_label = [p for p, i in zip(label_tiles, any_list) if i]

    print(f"{len(background_label)} background tiles, {len(signal_label)} signal tiles")

    # Separate the background only tiles.
    # Keep a fraction of background tiles.
    random_obj.shuffle(background_label)
    random_obj.shuffle(signal_label)
    print(f"With {len(signal_label)} signal tiles.")

    signal_labels = signal_label
    signal_images = [i.replace("labels", "images") for i in signal_labels]
    if keep_background_proportion < 1:
        keep_stop = int(keep_background_proportion * len(background_label))
        print(f"Keeping {keep_stop}/{len(background_label)} background tiles")
        background_label = background_label[:keep_stop]
    background_source = [i.replace("labels", "images") for i in background_label]

    if keep_signal_proportion < 1:
        keep_stop = int(keep_signal_proportion * len(signal_label))
        print(f"Keeping {keep_stop}/{len(signal_label)} signal tiles")
        signal_label = signal_label[:keep_stop]
    signal_source = [i.replace("labels", "images") for i in signal_label]

    return signal_labels, signal_images, background_label, background_source


def convert_mask(file):
    img = Image.open(file)
    thresh = 255
    fn = lambda x: 255 if x < thresh else 0
    out = img.convert("P").point(fn, mode="1")
    out = out.convert("P")
    palette = make_palette("dark", "light")
    out.putpalette(palette)
    out.save(file)


# train test val split
def train_test_split(file_list, test_size=0.1, val_size=0.1):
    random_obj.shuffle(file_list)
    train_size = 1 - test_size - val_size
    assert train_size > 0
    train_stop = int(len(file_list) * train_size)
    test_stop = int((len(file_list) * (train_size + test_size)))
    train_data = file_list[:train_stop]
    test_data = file_list[train_stop:test_stop]
    val_data = file_list[test_stop:]
    return train_data, test_data, val_data


# %%
if __name__ == "__main__":
    label_path = "dataset/labels"
    source_path = "dataset/images"
    training_area_path = "../data_220401/selected_area_220404.gpkg"
    keep_background_proportion = 1.0
    keep_signal_proportion = 1.0
    signal_labels, signal_images, bg_labels, bg_source = select_tiles(
        training_area_path, keep_signal_proportion, keep_background_proportion
    )
    print("signal label e.g.", signal_labels[0])
    print("background label e.g.", bg_labels[1])

    train_data, test_data, val_data = train_test_split(signal_labels)
    train_bg_data, test_bg_data, val_bg_data = train_test_split(bg_labels)

    output_folder = Path("dataset")

    for name, labels_paths in (
        ("training", train_data),
        ("evaluation", test_data),
        ("validation", val_data),
        ("training_bg", train_bg_data),
        # For evaluation and validation, put the signal and background together
        # in a single directory, no need to split.
        ("evaluation", test_bg_data),
        ("validation", val_bg_data),
    ):
        for label_path in labels_paths:
            img_path = label_path.replace("labels", "images")
            for i_name, i_path in (("labels", label_path), ("images", img_path)):
                location = output_folder / name / i_name
                dest = location / i_path[-20:]
                dest.parent.mkdir(exist_ok=True, parents=True)
                shutil.copy(i_path, dest)
                if i_name == "labels":
                    convert_mask(dest)
        print(name, "e.g.", dest)

    print("Successfully split dataset according to train-test-val")
