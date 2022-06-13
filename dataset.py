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


def load_img(label_path, source_path):
    print(label_path, source_path)
    signal_labels = [str(p) for p in Path(label_path).glob("*/*/*.png")]
    signal_images = [str(p) for p in Path(source_path).glob("*/*/*.png")]
    print(str(len(signal_labels)) + " label files found")
    print(str(len(signal_images)) + " source files found")
    return signal_labels, signal_images


def select_tiles(training_area_path, background_proportion=0):
    # Check for tiles inside the training area.
    # The training area label acts like another layer of labelling.
    training_area_list = [str(p) for p in Path(training_area_path).glob("*/*/*.png")]
    # Check which tiles are in the training area
    training_area_hasSignal = joblib.Parallel(n_jobs=4)(
        (
            joblib.delayed(lambda _p: cv2.imread(p).any())(p)
            for p in tqdm(training_area_list, desc="training area")
        )
    )
    training_area_list = [
        p for p, i in zip(training_area_list, training_area_hasSignal) if i
    ]

    # Find non-background tiles within the training area.
    training_area_list = [
        p.replace("training_area", "labels") for p in training_area_list
    ]
    training_area_list = [
        p for p in tqdm(training_area_list, desc="file exists") if Path(p).is_file()
    ]
    any_list = [
        cv2.imread(p).any() for p in tqdm(training_area_list, desc="backgrounds")
    ]
    if not any_list:
        raise ValueError("All tiles are background")
    background_tiles_list = [p for p, i in zip(training_area_list, any_list) if not i]
    signal_tiles_list = [p for p, i in zip(training_area_list, any_list) if i]

    # Separate the background only tiles.
    # Keep a fraction of background tiles.
    random.Random(123).shuffle(background_tiles_list)
    keep_stop = min(
        len(background_tiles_list), int(len(signal_tiles_list) * background_proportion)
    )
    print(f"Keeping {len(background_tiles_list)} background tiles")
    print(f"With {len(signal_tiles_list)} signal tiles.")

    signal_labels = signal_tiles_list
    signal_images = [i.replace("labels", "images") for i in signal_labels]
    background_label = background_tiles_list
    background_source = [i.replace("labels", "images") for i in background_label]

    return signal_labels, signal_images, background_label, background_source


def convert_mask(mask_list):
    for i in mask_list:
        img = Image.open(i)
        thresh = 0
        fn = lambda x: 255 if x > thresh else 0
        # values = np.unique(img.convert('P'))
        # print(values)
        out = img.convert("P").point(fn, mode="1")
        out = out.convert("P")
        palette = make_palette("dark", "light")
        out.putpalette(palette)
        out.save(i)
    print("Masks converted to 1bit labels, please check for correctness")


# train test val split
def train_test_split(file_list, test_size=0.2, val_size=0.2):
    random.Random(123).shuffle(file_list)
    train_size = 1 - test_size - val_size
    assert train_size > 0
    train_stop = int(len(file_list) * train_size)
    test_stop = int((len(file_list) * (train_size + test_size)))
    train_data = file_list[:train_stop]
    test_data = file_list[train_stop:test_stop]
    val_data = file_list[test_stop:]
    return train_data, test_data, val_data


if __name__ == "__main__":
    label_path = "dataset/labels"
    source_path = "dataset/images"
    training_area_path = "dataset/training_area"
    keep_background_tiles = 0.25
    signal_labels, signal_images, bg_label, bg_source = select_tiles(
        training_area_path, keep_background_tiles
    )
    convert_mask(signal_labels)
    convert_mask(bg_label)

    train_data, test_data, val_data = train_test_split(signal_labels)
    train_bg_data, test_bg_data, val_bg_data = train_test_split(bg_label)

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

    print("Successfully split dataset according to train-test-val")
