# %%
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

for s in ("training", "validation", "evaluation"):
    masks = list(Path(f"dataset/{s}/labels").glob("*/*/*png"))
    not_blank = [np.count_nonzero(cv2.imread(str(p)) > 200) for p in tqdm(masks)]
    count_not_blank = np.sum(not_blank) # number of non-blank pixels
    all_not_blank = np.count_nonzero(not_blank) # number of non blank tiles
    n = len(masks)
    print(f"{s}: all {n}, all pix {all_pix}, not blank {count_not_blank}, all not blank {all_not_blank}")
    pos_wt = 1-count_not_blank/(all_not_blank*2)/(256**2)
    print(f"pos wt = {pos_wt:0.2e}, neg_wt = {1-pos_wt:0.2e}")

    # img = cv2.imread(str(masks[0]))
    # print(img)
    # breakpoint()

# %%
