from pathlib import Path
from dataset import convert_mask
from tqdm import tqdm

for s in ("training", "validation", "evaluation"):
    masks = (str(p) for p in tqdm(list(Path(f"dataset/{s}/labels").glob("*/*/*png"))))
    convert_mask(masks)
