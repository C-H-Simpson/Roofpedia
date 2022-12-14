"""
Check the amount of positive and negative pixels / examples in each dataset.
"""
# %%
from pathlib import Path

import numpy as np
from PIL import Image


def get_signal_weight(input_glob):
    masks = list(input_glob)
    assert len(masks)
    not_blank = [np.count_nonzero(Image.open(str(p))) for p in (masks)]
    count_not_blank = np.sum(not_blank)  # number of non-blank pixels
    all_not_blank = np.count_nonzero(not_blank)  # number of non blank tiles
    n = len(masks)
    print(f"\t{n=}, {count_not_blank=}, {all_not_blank=}")
    pos_wt = 1 - count_not_blank / (all_not_blank * 2) / (256 ** 2)
    print(f"\tpos wt = {pos_wt:0.2e}, neg_wt = {1-pos_wt:0.2e}")
    return pos_wt, 1 - pos_wt


def count_signal_pixels(input_glob):
    masks = list(input_glob)
    assert len(masks)
    not_blank = [np.count_nonzero(Image.open(str(p))) for p in (masks)]
    count_not_blank = np.sum(not_blank)  # number of non-blank pixels
    all_not_blank = np.count_nonzero(not_blank)  # number of non blank tiles
    n = len(masks)
    return {
        "n_tiles": n,
        "n_signal_pixels": count_not_blank,
        "n_signal_tiles": all_not_blank
    }


if __name__ == "__main__":
    dataset_parents = list(Path("dataset").glob("*"))
    assert dataset_parents
    for p in dataset_parents:
        for s in ("training_s", "training_b", "validation"):
            if not (p / s).is_dir():
                # print(p / s, "not a dir")
                continue
            get_signal_weight((p / s).glob("labels/*/*png"))