"""
Identify the source imagery and labelling.
Split it for k-fold validation.

NB, there are two sets of imagery we are using.
We treat the imagery collected in 2021 as the "base" imagery.
We then use the 2019 imagery as an additional validation/testing dataset.
The two different imagery sets use the same tiling, so tiles with the same 
name need to be selected.
"""
# %%
import random
import shutil
from os import symlink
from pathlib import Path

import geopandas as gpd
import numpy as np
from PIL import Image
from tqdm import tqdm

from imagery_tiling.batched_tiling import tiling_path
from src.colors import make_palette

random_obj = random.Random(144)

# This is the location from which the imagery will be taken.
# This has been created by imagery_tiling/batched_tiling.py
source_path = "/home/ucbqc38/Scratch/getmapping_2021_tiled/"
# This is the number of splits.
k_folds = 5
# This is a geodata file that labels the area that was hand labelled.
training_area_path = "../data/selected_area_220404.gpkg"

# Get building footprints
buildings = gpd.read_feather("../data/os_buildings_2021.feather")

keep_background_proportion = 1
keep_signal_proportion = 1.0

dataset_folder = Path("dataset")


def convert_mask(file):
    img = Image.open(file)
    out = img.convert("P").point(lambda x: 255 if x > 0 else 0, mode="1")
    out = out.convert("P")
    palette = make_palette("dark", "light")
    out.putpalette(palette)
    out.save(file)


def kfold_split(file_list, n_splits=5):
    random_obj.shuffle(file_list)
    splits = []
    N = len(file_list)
    split_size = int(N / n_splits)
    for i_split in range(n_splits):
        split_start = i_split * split_size
        split_stop = min(split_start + split_size, N)
        splits.append(file_list[split_start:split_stop])
    return splits


# %%
# Kfold
if __name__ == "__main__":
    # Delete the old dataset, otherwise it piles up.
    assert not dataset_folder.exists(), "recommend deleting the folder first"

    # Load the file that encodes the geometry of the tiling.
    gdf_tiles = gpd.read_feather(tiling_path)

    # Load the limits of the labelled area.
    gdf_labelled_area = gpd.read_file(training_area_path)

    # %%
    # Limit to the valid labelled area.
    intersect = (
        # This version excludes tiles at the edge of the domain, reducing the amount of data available.
        gdf_tiles[gdf_tiles.within(gdf_labelled_area.unitary_union)]
        # This version has a slight problem because tiles at the edge of the labelled area will be included.
        # gpd.overlay(gdf_tiles, gdf_labelled_area).dissolve(["x", "y"]).reset_index()
    )
    print(f"{len(intersect)} in labelled area")
    # Only keep tiles that intersect with a building.
    intersect = (
        gpd.overlay(intersect, buildings).dissolve(["x", "y"]).reset_index()
    )
    print(f"{len(intersect)} with buildings")

    # Turn these into paths
    intersect = intersect.assign(
        p=intersect.apply(
            lambda _df: Path(source_path)
            / _df.TILE_NAME
            / "images"
            / str(int(_df.x))
            / (str(int(_df.y)) + ".png"),
            axis=1,
        )
    )
    # Check at least some files exist
    assert intersect.apply(lambda _df: ((_df.p).is_file()), axis=1).any()

    # %%
    # Drop tiles with no labels
    image_tiles = [p for p in intersect.p if p.is_file()]
    label_tiles = [Path(str(p).replace("images", "labels")) for p in image_tiles]
    # image_tiles = [p for p, pi in zip(image_tiles, label_tiles) if pi.is_file()] # There should be none of these
    label_tiles = [p for p in label_tiles if p.is_file()]

    assert len(image_tiles)
    assert len(label_tiles) == len(image_tiles)

    # %%
    print(f"{len(image_tiles)} tiles in training area")

    # %%
    # Find non-background tiles within the training area.
    any_list = [
        np.array(Image.open(p)).any() for p in tqdm(label_tiles, desc="backgrounds")
    ]
    print(f"Positive labels exist for {sum(any_list)} tiles")
    if not any_list:
        raise ValueError("All tiles are background")

    # %%
    background_label = [p for p, i in zip(label_tiles, any_list) if not i]
    signal_label = [p for p, i in zip(label_tiles, any_list) if i]

    print(f"{len(background_label)} background tiles, {len(signal_label)} signal tiles")

    # %%
    # Shuffle
    random_obj.shuffle(background_label)
    random_obj.shuffle(signal_label)

    # %%
    # Separate the background only tiles.
    # Keep a fraction of background tiles.
    signal_labels = signal_label
    signal_images = [str(i).replace("labels", "images") for i in signal_labels]
    if keep_background_proportion < 1:
        keep_stop = int(keep_background_proportion * len(background_label))
        print(f"Keeping {keep_stop}/{len(background_label)} background tiles")
        background_label = background_label[:keep_stop]
    background_source = [str(i).replace("labels", "images") for i in background_label]

    # %%
    if keep_signal_proportion < 1:
        keep_stop = int(keep_signal_proportion * len(signal_label))
        print(f"Keeping {keep_stop}/{len(signal_label)} signal tiles")
        signal_label = signal_label[:keep_stop]

    # %%
    s_labels, s_images, b_labels, b_source = (
        signal_labels,
        signal_images,
        background_label,
        background_source,
    )
    print("s label e.g.", s_labels[0])
    print("background label e.g.", b_labels[1])

    # %%
    s_splits = kfold_split(s_labels)
    b_splits = kfold_split(b_labels)

    # %%
    # Split the files into directories.
    for k in range(k_folds):
        s = s_splits[k]
        b = b_splits[k]
        for name, labels_paths in ((f"{k}s", s), (f"{k}b", b)):
            for label_path in labels_paths:
                img_path = str(label_path).replace("labels", "images")
                for i_name, i_path in (("labels", label_path), ("images", img_path)):
                    i_path = Path(i_path)
                    location = dataset_folder / name / i_name
                    dest = location / i_path.parent.stem / i_path.name
                    dest.parent.mkdir(exist_ok=True, parents=True)
                    shutil.copy(i_path, dest)
                    if i_name == "labels":
                        convert_mask(dest)
            print(name, "e.g.", dest)

    # %%
    # Add in alternative imagery set.
    for k in range(k_folds):
        s = s_splits[k]
        b = b_splits[k]
        for name, labels_paths in ((f"{k}s_alt", s), (f"{k}b_alt", b)):
            for label_path in labels_paths:
                # Use 2019 imagery as the alternative
                label_path = Path(
                    str(label_path).replace("getmapping_2021", "getmapping_2019")
                )
                img_path = Path(str(label_path).replace("labels", "images"))
                if not (img_path.is_file() and label_path.is_file()):
                    continue
                for i_name, i_path in (("labels", label_path), ("images", img_path)):
                    i_path = Path(i_path)
                    location = dataset_folder / name / i_name
                    dest = location / i_path.parent.stem / i_path.name
                    dest.parent.mkdir(exist_ok=True, parents=True)
                    shutil.copy(i_path, dest)
                    if i_name == "labels":
                        convert_mask(dest)
            print(name, "e.g.", dest)

    # %%
    # Remove the "testing" folder if it exists.
    # if (dataset_folder/"testing").exists():
    # shutil.rmtree(str(dataset_folder / "testing"))

    # %%
    # Identify fold i=0 as the test data
    testing_folders = list((dataset_folder / "0s").glob("*/*/*png")) + list(
        (dataset_folder / "0b").glob("*/*/*png")
    )
    dest = dataset_folder / "testing"
    for f in testing_folders:
        dest_file = dest / f.parent.parent.stem / f.parent.stem / f.name
        dest_file.parent.mkdir(exist_ok=True, parents=True)
        shutil.move(str(f), str(dest_file))

    testing_folders = list((dataset_folder / "0s_alt").glob("*/*/*png")) + list(
        (dataset_folder / "0b_alt").glob("*/*/*png")
    )
    dest = dataset_folder / "testing_alt"
    for f in testing_folders:
        dest_file = dest / f.parent.parent.stem / f.parent.stem / f.name
        dest_file.parent.mkdir(exist_ok=True, parents=True)
        shutil.move(str(f), str(dest_file))

    # %%
    # At this point we have folders that look like
    # 1b, 1s, 1b_alt, 1s_alt, 2b etc.
    # We need to link these into one folder for each fold - one for training_s, training_b, validation, validation_alt

    # %%
    # Use softlinks to assemble training / validation sets from the other folds.
    # needs to have a structure like k1/(training|training_bg)/images/19/....
    for k in range(1, k_folds):
        print(f"{k=}")
        for i in range(1, k_folds):
            print(f"{i=}")
            for s_or_b in ("s", "b", "s_alt", "b_alt"):
                print(f"{s_or_b=}")
                # For i=k, the folders {i}b/{i}s contain the validation data
                # and others are training data
                if i == k:
                    name = "validation"
                    if "alt" in s_or_b:
                        name = "validation_alt"
                    destination = dataset_folder / f"k{k}" / name
                else:
                    name = "training"
                    destination = dataset_folder / f"k{k}" / f"training_{s_or_b}"

                labels_paths = list(dataset_folder.glob(f"{i}{s_or_b}/labels/*/*.png"))
                for p in labels_paths:

                    dest_file = destination / "labels" / p.parent.stem / p.name
                    dest_file.parent.mkdir(exist_ok=True, parents=True)
                    symlink(p.resolve(), dest_file.resolve())
                    # Now repeat for the image
                    p = Path(str(p).replace("labels", "images"))
                    dest_file = destination / "images" / p.parent.stem / p.name
                    dest_file.parent.mkdir(exist_ok=True, parents=True)
                    symlink(p.resolve(), dest_file.resolve())

                print("example", (p.resolve(), dest_file))
# %%
