from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from src.colors import make_palette
from dataset import convert_mask
from tqdm import tqdm

for s in ("training", "validation", "evaluation"):
    masks = list(Path(f"dataset/{s}/labels").glob("*/*/*png"))
    not_blank = [np.count_nonzero(cv2.imread(str(p))>200) for p in tqdm(masks)]
    count_not_blank = np.sum(not_blank)
    all_not_blank = np.count_nonzero(not_blank)
    n = len(masks)
    print(f"{s}: all {n}, not blank {count_not_blank}, count not blank {all_not_blank}")

    #img = cv2.imread(str(masks[0])) 
    #print(img)
    #breakpoint()
