"""
Check the amount of positive and negative pixels / examples in each dataset.
"""
# %%
from pathlib import Path

from PIL import Image
import numpy as np

def get_signal_weight(dir: Path):
    masks = list((dir / "labels").glob("*/*/*png"))
    not_blank = [np.count_nonzero(Image.open(str(p))) for p in (masks)]
    count_not_blank = np.sum(not_blank)  # number of non-blank pixels
    all_not_blank = np.count_nonzero(not_blank)  # number of non blank tiles
    n = len(masks)
    print(f"\t{n=}, {count_not_blank=}, {all_not_blank=}")
    pos_wt = 1 - count_not_blank / (all_not_blank * 2) / (256 ** 2)
    print(f"\tpos wt = {pos_wt:0.2e}, neg_wt = {1-pos_wt:0.2e}")
    return pos_wt, 1-pos_wt


if __name__ == "__main__":
    dataset_parents = Path("dataset").glob("*")
    for p in dataset_parents:
        for s in ("training_s", "training_b", "validation"):
            if not (p / s).is_dir():
                # print(p / s, "not a dir")
                continue
            get_signal_weight(p/s)
# %%
