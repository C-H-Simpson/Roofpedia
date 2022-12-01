# %%
from pathlib import Path
import shutil

files = {}
tiles = {}
bounds = {}
boxes = {}

ds = "evaluation"
files[ds] = list(Path(f"dataset/{ds}/labels/19").glob("*/*.png"))
tiles[ds] = [
    # Directory structure is z/x/y
    (int(s.parent.stem), int(s.stem), int(s.parent.parent.stem))
    for s in files[ds]
]

# %%
equivalent_imageset = Path("../data_220922/mapbox_evaluation")
destination = Path("dataset/evaluation_alternative")
destination.mkdir(exist_ok=True)

# %%
for t in files[ds]:
    stem = Path(t.parent.parent.stem) / t.parent.stem / t.name
    if not t.is_file():
        print(f"{t} does not exist")
        continue
    # print(equivalent_imageset / stem, destination / stem)
    (destination / stem).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(equivalent_imageset / stem, destination / stem)
