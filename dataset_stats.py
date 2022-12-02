"""
Check the amount of positive and negative pixels / examples in each dataset.
"""
from pathlib import Path

import cv2
import numpy as np

dataset_parents = Path("dataset").glob("*")
for p in dataset_parents:
    for s in ("training_s", "training_b", "validation"):
        if not (p / s).is_dir():
            # print(p / s, "not a dir")
            continue
        masks = list((p / s / "labels").glob("*/*/*png"))
        not_blank = [np.count_nonzero(cv2.imread(str(p)) > 200) for p in (masks)]
        count_not_blank = np.sum(not_blank)  # number of non-blank pixels
        all_not_blank = np.count_nonzero(not_blank)  # number of non blank tiles
        n = len(masks)
        print(p / s)
        print(f"\t{n=}, {count_not_blank=}, {all_not_blank=}")
        pos_wt = 1 - count_not_blank / (all_not_blank * 2) / (256**2)
        print(f"\tpos wt = {pos_wt:0.2e}, neg_wt = {1-pos_wt:0.2e}")

        # img = cv2.imread(str(masks[0]))
        # print(img)
        # breakpoint()
