"""
Based on the each split in the 2021 data, get equivalent data for 2019.
"""
# %%
import shutil
from pathlib import Path

from tqdm import tqdm

from imagery_tiling.batched_tiling import tiling_path

# %%
dataset = list(Path("dataset").glob("k*/*/images/*/*png")) + list(Path("dataset").glob("*/images/*/*png"))
alt_source = Path("/home/ucbqc38/Scratch/getmapping_2019_tiled")
alt_destination = Path("alt_dataset")
alt_destination.mkdir()

# %%
for p in tqdm(dataset):
    p_label = Path(str(p).replace("images", "labels"))
    x, y = float(p.parent.stem), float(p.stem)
    alt = list(alt_source.glob(f"*/images/{x:0.0f}/{y:0.0f}.png"))
    assert len(alt) == 1, p
    alt = alt[0]
    alt_label = Path(str(alt).replace("images", "labels"))
    assert alt_label.exists()
    destination = Path(str(p).replace("dataset", "alt_dataset"))
    destination_label = Path(str(p_label).replace("dataset", "alt_dataset"))
    destination.parent.mkdir(exist_ok=True, parents=True)
    destination_label.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy(alt, destination)
    shutil.copy(alt_label, destination_label)

# %%
# May need to do this if you didn't construct the masks properly!
# But don't run this part repeatedly.
# from dataset import convert_mask
# from pathlib import Path

# g = list(Path("alt_dataset").glob("k*/*/labels/*/*png"))# + list(Path("dataset/testing_alt/labels/*/*png"))
# [convert_mask(f) for f in g]