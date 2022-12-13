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

keep_background_proportion = 0.01
keep_signal_proportion = 1.0

# %%
def load_img(label_path, source_path):
    print(label_path, source_path)
    signal_labels = [str(p) for p in Path(label_path).glob("*/*/*.png")]
    signal_images = [str(p) for p in Path(source_path).glob("*/*/*.png")]
    print(str(len(signal_labels)) + " label files found")
    print(str(len(signal_images)) + " source files found")
    return signal_labels, signal_images


def convert_mask(file):
    img = Image.open(file)
    thresh = 255
    out = img.convert("P").point(lambda x: 255 if x < thresh else 0, mode="1")
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
    # Load the file that encodes the geometry of the tiling.
    gdf_tiles = gpd.read_feather(tiling_path)

    # Load the limits of the labelled area.
    gdf_labelled_area = gpd.read_file(training_area_path)

    # %%
    # Limit to the valid labelled area.
    intersect = (
        gpd.overlay(gdf_tiles, gdf_labelled_area).dissolve(["x", "y"]).reset_index()
    )

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
    image_tiles = [p for p, pi in zip(image_tiles, label_tiles) if pi.is_file()]
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
    i_path.name

    # %%
    s_splits = kfold_split(s_labels)
    b_splits = kfold_split(b_labels)

    output_folder = Path("dataset")

    for k in range(k_folds):
        s = s_splits[k]
        b = b_splits[k]
        for name, labels_paths in ((f"{k}s", s), (f"{k}b", b)):
            for label_path in labels_paths:
                img_path = str(label_path).replace("labels", "images")
                for i_name, i_path in (("labels", label_path), ("images", img_path)):
                    i_path = Path(i_path)
                    location = output_folder / name / i_name
                    dest = location / i_path.parent.stem / i_path.name
                    dest.parent.mkdir(exist_ok=True, parents=True)
                    shutil.copy(i_path, dest)
                    if i_name == "labels":
                        convert_mask(dest)
            print(name, "e.g.", dest)

    # %%
    Image.open("dataset/0s/images/532304/180043.png")

    # %%
    Image.open("dataset/0s/labels/532304/180043.png")

    # %%
    i_path
    # %%
    np.array(Image.open("/home/ucbqc38/Scratch/getmapping_2021_tiled/TQ38/labels/532304/180043.png")).shape

    # %%
    Path("/home/ucbqc38/Scratch/getmapping_2021_tiled/TQ38/labels/532304/180043.png.aux.xml").read_text()
    # %%
    Path("/home/ucbqc38/Scratch/getmapping_2021_tiled/TQ38/images/532304/180043.png.aux.xml").read_text()

    # %%
    np.array(Image.open(str(i_path).replace("images", "labels"))).shape

    # %%
    # Identify fold 0 as the test data
    shutil.move(output_folder / "0s", output_folder / "testing")
    shutil.move(output_folder / "0b", output_folder / "testing")

    # %%
    # Use softlinks to assemble training / validation sets from the other folds.
    # needs to have a structure like k1/(training|training_bg)/images/19/....
    training_s_labels_paths = {
        k: list((output_folder / f"{k}s" / "labels").glob("*/*/*png"))
        for k in range(k_folds)
    }
    training_b_labels_paths = {
        k: list((output_folder / f"{k}b" / "labels").glob("*/*/*png"))
        for k in range(k_folds)
    }
    for k in range(1, k_folds):
        sel = list(range(1, k)) + list(range(k + 1, k_folds))
        # print(k, sel)
        for i in range(1, k_folds):
            for name, labels_paths in (
                ("training_s", training_s_labels_paths[i]),
                ("training_b", training_b_labels_paths[i]),
            ):
                if i == k:
                    name = "validation"
                print(name)
                dest_folder = output_folder / f"k{k}" / name
                for p in labels_paths:
                    dest_file = (
                        dest_folder
                        / p.parent.parent.parent.stem
                        / p.parent.parent.stem
                        / p.parent.stem
                        / p.name
                    )
                    print(p, dest_file)
                    if not p.is_file():
                        raise ValueError()
                    dest_file.parent.mkdir(exist_ok=True, parents=True)
                    symlink(p.resolve(), dest_file)
                    p = Path(str(p).replace("labels", "images"))
                    if not p.is_file():
                        raise ValueError()
                    dest_file = Path(str(dest_file).replace("labels", "images"))
                    dest_file.parent.mkdir(exist_ok=True, parents=True)
                    print(p, dest_file)
                    symlink(p.resolve(), dest_file)

    print("Successfully split dataset according to kfold split")