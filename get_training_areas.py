"""
Make a geojson showing what areas are in each dataset.
"""
# %%
from pathlib import Path

import geopandas as gpd

# from dataset import k_folds
# from imagery_tiling.batched_tiling import tiling_path
tiling_path = "./tiling_256_0.25.feather"
k_folds = 4

gdf = gpd.read_feather(tiling_path).set_index(["x", "y"])

files = {}
tiles = {}
bounds = {}
boxes = {}
for k in range(1, k_folds):
    for ds in ("training_s", "validation", "testing", "training_b"):
        files[ds] = list(Path(f"dataset/k{k}/{ds}/images").glob("*/*.png"))
        tiles[ds] = [
            # Directory structure is x/y
            (int(s.parent.stem), int(s.stem))
            for s in files[ds]
        ]
        bounds[ds] = gdf.loc[tiles[ds]].assign(ds=ds)
        gdf[["geometry"]].to_file(f"dataset/k{k}/{ds}.geojson", driver="GeoJSON", index=False)
