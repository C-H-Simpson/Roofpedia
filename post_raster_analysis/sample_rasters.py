# %%
import geopandas as gpd
import rasterio as rio
import numpy as np
from rasterio import mask
from pathlib import Path
import matplotlib.pyplot as plt

native_crs = "EPSG:27700"

# %%
# The sample areas
sample = gpd.read_file("../labels_rename/selected_area_220404.gpkg").to_crs(native_crs)

# Get the 1km grid references
osgb_1km = gpd.read_file(
    "../../GIS/OSGB_grids/Shapefile/OSGB_Grid_1km.shp"
).to_crs(native_crs)

# %%
# For each 1km tile intersecting with the predictions.
# Sample the raster imagery into a flat N by 3 array.
year = 2019
predictions = gpd.read_file(f"results/buffered_polygons_{year}_simplify.geojson").to_crs(native_crs)
predictions = gpd.overlay(predictions, sample, "intersection") # Select only in sample area
labels = gpd.read_file(f"../labels_rename/gr_manual_labels_checked_{year}.geojson")
labels = gpd.GeoDataFrame(geometry=list(labels.geometry.buffer(0).unary_union.geoms)) # Fix self-intersection
labels = gpd.overlay(labels, sample, "intersection") # Select only in sample area
tiles = set(predictions["PLAN_NO"]).union(set(labels["PLAN_NO"]))
# %%
conf = dict(
    tp = gpd.overlay(predictions[["geometry"]], labels[["geometry"]], "intersection"),
    fp = gpd.overlay(predictions[["geometry"]], labels[["geometry"]], "difference"),
    fn = gpd.overlay(labels[["geometry"]], predictions[["geometry"]], "difference"),
)

# %%
# imagery_glob = list( Path(f"/home/ucbqc38/Scratch/getmapping_{year}").glob("*/*/*jpg"))
imagery_glob = list( Path(f"../../GIS/getmapping_latest_london_imagery").glob("*/*/*/*jpg"))
assert imagery_glob
# Create a dict for the input imagery paths to the 1km grid references.
imagery_path = {g.stem[0:6].upper(): g for g in imagery_glob}

# %%
results = {}
for name in conf:
    results[name] = np.array([[],[],[]], dtype="uint8")
    shapes = list(conf[name].geometry)
    for tile in tiles:
        with rio.open(imagery_path[tile], 'r') as src:
            out_image, out_transform = mask.mask(src, shapes, crop=True)

        out_image = out_image.reshape(3, -1)
        out_mask = out_image.any(0)
        out_image = out_image[:, out_mask]
        results[name] = np.concatenate((results[name], out_image), axis=1)

        if out_mask.any():
            break

# %%
out_image.dtype
# %%
for name in results:
    ness = results[name] / results[name].mean(0)
    plt.hist(ness.T, color=("r", "g", "b"), label=name, bins=np.linspace(0.5, 1.5, 10), histtype="step")
    print(name, ness.mean(1))
plt.legend()
# %%
for name in results:
    ness = results[name] / results[name].mean(0)
    plt.scatter(ness[0], ness[1], label=name, alpha=0.5)
plt.legend()

# %%
# False positives in this dataset tend to be more red rather than green.
fig, ax = plt.subplots(2)
for name in results:
    for i in (0, 1,):
        ax[i].hist(results[name][i], label=name, bins=np.arange(40, 220, 1), histtype="step", density=True)
ax[0].set_xlabel("Red")
ax[1].set_xlabel("Green")
plt.legend()
plt.tight_layout()

# %%
# False positives in this dataset tend to be more red rather than green.
fig, ax = plt.subplots(4, figsize=(6, 6))
for name in results:
    ness = results[name] / results[name].mean(0)
    for i in (0, 1, 2):
        ax[i].hist(ness[i], label=name, bins=np.linspace(0.8, 1.4, 100), histtype="step", density=True)
    ax[3].hist(results[name].mean(0), label=name, bins=np.arange(0, 256, 2), histtype="step", density=True)
ax[0].set_xlabel("Redness")
ax[1].set_xlabel("Greenness")
ax[2].set_xlabel("Blueness")
ax[3].set_xlabel("Brightness")
plt.legend()
plt.tight_layout()

# %%
results[name].mean(0)