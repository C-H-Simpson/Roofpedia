import argparse
import glob
import os
import random
import shutil
from pathlib import Path

import cv2
import joblib
import numpy as np
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
    # Check for tiles inside the training area.
    # The training area label acts like another layer of labelling.
    training_area_list = [str(p) for p in Path(training_area_path).glob("*/*/*.png")]
    print(f"{len(training_area_list)} tiles in training extent")
    # Check which tiles are in the training area
    in_training_area = joblib.Parallel(n_jobs=4)(
        (
            joblib.delayed(lambda _p: not cv2.imread(p).any())(p)
            for p in tqdm(training_area_list, desc="training area")
        )
    )
    training_area_list = [p for p, i in zip(training_area_list, in_training_area) if i]
    print(f"{len(training_area_list)} tiles in training area")

    # Find non-background tiles within the training area.
    training_area_list = [
        p.replace("training_area", "labels") for p in training_area_list
    ]
    training_area_list = [
        p for p in tqdm(training_area_list, desc="file exists") if Path(p).is_file()
    ]
    print(f"Labels exist for {len(training_area_list)} tiles")
    any_list = [
        cv2.imread(p).any() for p in tqdm(training_area_list, desc="backgrounds")
    ]
    if not any_list:
        raise ValueError("All tiles are background")
    background_label = [p for p, i in zip(training_area_list, any_list) if not i]
    signal_label = [p for p, i in zip(training_area_list, any_list) if i]

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
    training_area_path = "dataset/training_area"
    keep_background_proportion = 0.01  # 0.1
    keep_signal_proportion = 1.0  # 0.1
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
                dest = location / label_path[-20:]
                dest.parent.mkdir(exist_ok=True, parents=True)
                shutil.copy(label_path, dest)
                convert_mask(dest)
        print(name, "e.g.", dest)

    print("Successfully split dataset according to train-test-val")
