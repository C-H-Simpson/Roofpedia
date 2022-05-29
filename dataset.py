import glob
import numpy as np
import os
import shutil
from PIL import Image
import random
import cv2 
from src.colors import make_palette
import argparse
from pathlib import Path
from tqdm import tqdm
os.getcwd()

def load_img(target_path, source_path):
    print(target_path, source_path)
    files_target = [str(p) for p in Path(target_path).glob('*/*/*.png')]
    files_source = [str(p) for p in Path(source_path).glob('*/*/*.png')]
    print(str(len(files_target)) + ' target files found')
    print(str(len(files_source)) + ' source files found')
    return files_target, files_source

def select_tiles(training_area_path, blank_proportion=0):
    # Check for tiles inside the training area.
    # The training area label acts like another layer of labelling.
    training_area_list = [str(p) for p in Path(training_area_path).glob("*/*/*.png")]
    # Check which tiles are in the training area
    training_area_list = [p for p in tqdm(training_area_list, desc="training area") if cv2.imread(p).any()]

    # Find non-blank tiles within the training area.
    training_area_list = [p.replace("training_area", "labels") for p in training_area_list]
    training_area_list = [p for p in tqdm(training_area_list, desc="file exists") if Path(p).is_file()]
    any_list = [cv2.imread(p).any() for p in tqdm(training_area_list, desc="blanks")]
    if not any_list:
        raise ValueError("All tiles are blank")
    blank_tiles_list = [p for p, i in zip(training_area_list, any_list) if not i]
    notblank_tiles_list = [p for p, i in zip(training_area_list, any_list) if i]

    # Keep a fraction of blank tiles.
    random.Random(123).shuffle(blank_tiles_list)
    keep_stop = min(len(blank_tiles_list), int(len(notblank_tiles_list)*blank_proportion))
    print(f"Keeping {keep_stop} of {len(blank_tiles_list)} blank tiles")
    print(f"With {len(notblank_tiles_list)} not blank tiles.")
    blank_tiles_list = blank_tiles_list[:keep_stop]
    files_target = notblank_tiles_list + blank_tiles_list

    files_source = [i.replace("labels", "images") for i in files_target]

    return files_target, files_source


def convert_mask(mask_list):
    for i in mask_list:
        img = Image.open(i)
        thresh = 0
        fn = lambda x : 255 if x > thresh else 0
        #values = np.unique(img.convert('P'))
        #print(values)
        out = img.convert('P').point(fn, mode='1')
        out = out.convert('P')
        palette = make_palette("dark", "light")
        out.putpalette(palette)
        out.save(i)
    print("Masks converted to 1bit labels, please check for correctness")
# train test val split
def train_test_split(file_list, test_size = 0.2, val_size=0.2):
    random.Random(123).shuffle(file_list)
    train_size = 1 - test_size - val_size
    assert train_size>0
    train_stop = int(len(file_list)*train_size)
    test_stop = int((len(file_list) * (train_size+test_size)))
    train_data = file_list[:train_stop]
    test_data = file_list[train_stop: test_stop]
    val_data = file_list[test_stop:]
    return train_data, test_data, val_data


if __name__ == "__main__":
    target_path = 'dataset/labels'
    source_path = 'dataset/images'
    training_area_path = 'dataset/training_area'
    keep_blank_tiles = 0.25
    files_target, files_source = select_tiles(training_area_path, keep_blank_tiles)
    convert_mask(files_target)

    train_data, test_data, val_data = train_test_split(files_target)
    train_data_img = []
    test_data_img =[]
    val_data_img =[]
    
    output_folder = 'dataset'
    
    for i in train_data:
        if not os.path.exists(output_folder +'/training/labels/'):
            os.makedirs(output_folder +'/training/labels/')
        dest = output_folder +'/training/labels/' + i[-20:]
        if not os.path.exists(os.path.dirname(dest)):
            os.makedirs(os.path.dirname(dest))
        shutil.copy(i, dest)

    for i in test_data:
        if not os.path.exists(output_folder +'/evaluation/labels/'):
            os.makedirs(output_folder +'/evaluation/labels/')
        dest = output_folder +'/evaluation/labels/' + i[-20:]
        if not os.path.exists(os.path.dirname(dest)):
            os.makedirs(os.path.dirname(dest))
        shutil.copy(i, dest)

    for i in val_data:
        if not os.path.exists(output_folder +'/validation/labels/'):
            os.makedirs(output_folder +'/validation/labels/')
        dest = output_folder +'/validation/labels/' + i[-20:]
        if not os.path.exists(os.path.dirname(dest)):
            os.makedirs(os.path.dirname(dest))
        shutil.copy(i, dest)

    for i in train_data:
        train_data_img.append(i.replace('labels', 'images')) 
    for i in test_data:
        test_data_img.append(i.replace('labels', 'images')) 
    for i in val_data:
        val_data_img.append(i.replace('labels', 'images')) 

    for i in train_data_img:
        if not os.path.exists(output_folder +'/training/images/'):
            os.makedirs(output_folder +'/training/images/')
        dest = output_folder +'/training/images/' + i[-20:]
        if not os.path.exists(os.path.dirname(dest)):
            os.makedirs(os.path.dirname(dest))
        shutil.copy(i, dest)

    for i in test_data_img:
        if not os.path.exists(output_folder +'/evaluation/images/'):
            os.makedirs(output_folder +'/evaluation/images/')
        dest = output_folder +'/evaluation/images/' + i[-20:]
        if not os.path.exists(os.path.dirname(dest)):
            os.makedirs(os.path.dirname(dest))
        shutil.copy(i, dest)

    for i in val_data_img:
        if not os.path.exists(output_folder +'/validation/images/'):
            os.makedirs(output_folder +'/validation/images/')
        dest = output_folder +'/validation/images/' + i[-20:]
        if not os.path.exists(os.path.dirname(dest)):
            os.makedirs(os.path.dirname(dest))
        shutil.copy(i, dest)

    print("Successfully split dataset according to train-test-val")

