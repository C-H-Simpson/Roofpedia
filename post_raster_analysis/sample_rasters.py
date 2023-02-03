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
osgb_1km = gpd.read_file("../../GIS/OSGB_grids/Shapefile/OSGB_Grid_1km.shp").to_crs(
    native_crs
)

# %%
# For each 1km tile intersecting with the predictions.
# Sample the raster imagery into a flat N by 3 array.
year = 2019
predictions = gpd.read_file(f"results/buffered_polygons_{year}.geojson").to_crs(
    native_crs
)
predictions = gpd.overlay(
    predictions, sample, "intersection"
)  # Select only in sample area
labels = gpd.read_file(f"../labels_rename/gr_manual_labels_checked_{year}.geojson")
labels = gpd.GeoDataFrame(
    geometry=list(labels.geometry.buffer(0).unary_union.geoms)
)  # Fix self-intersection
labels = gpd.overlay(labels, sample, "intersection")  # Select only in sample area
tiles = set(predictions["PLAN_NO"]).union(set(labels["PLAN_NO"]))
# %%
conf = dict(
    TP=gpd.overlay(predictions[["geometry"]], labels[["geometry"]], "intersection"),
    FP=gpd.overlay(predictions[["geometry"]], labels[["geometry"]], "difference"),
    FN=gpd.overlay(labels[["geometry"]], predictions[["geometry"]], "difference"),
)

# %%
# imagery_glob = list( Path(f"/home/ucbqc38/Scratch/getmapping_{year}").glob("*/*/*jpg"))
imagery_glob = list(
    Path(f"../../GIS/getmapping_latest_london_imagery").glob("*/*/*/*jpg")
)
assert imagery_glob
# Create a dict for the input imagery paths to the 1km grid references.
imagery_path = {g.stem[0:6].upper(): g for g in imagery_glob}

# %%
results = {}
for name in conf:
    results[name] = np.array([[], [], []], dtype="uint8")
    shapes = list(conf[name].geometry)
    total_size = 0
    for tile in tiles:
        with rio.open(imagery_path[tile], "r") as src:
            size = src.shape[0] * src.shape[1]
            total_size += size
            try:
                out_image, out_transform = mask.mask(src, shapes, crop=True)
            except ValueError:
                continue

        out_image = out_image.reshape(3, -1)
        out_mask = out_image.any(0)
        out_image = out_image[:, out_mask]
        results[name] = np.concatenate((results[name], out_image), axis=1)


# %%
fig, ax = plt.subplots(3)
for name in results:
    for i in (0, 1, 2):
        ax[i].hist(
            results[name][i], label=name, bins=np.arange(40, 220, 1), histtype="step"
        )
ax[0].set_xlabel("Red")
ax[1].set_xlabel("Green")
ax[2].set_xlabel("Blue")
plt.legend()
plt.tight_layout()

# %%
# False positives in this dataset tend to be more red rather than green, but overlap with the TP.
fig, ax = plt.subplots(4, figsize=(6, 6))
for name in results:
    ness = results[name] / (results[name].mean(0)) / 3
    for i in (0, 1, 2):
        ax[i].hist(
            ness[i], label=name, bins=np.linspace(0.25, 0.40, 50), histtype="step"
        )
    ax[3].hist(
        results[name].mean(0), label=name, bins=np.arange(50, 200, 5), histtype="step"
    )
ax[0].set_xlabel("Redness")
ax[1].set_xlabel("Greenness")
ax[2].set_xlabel("Blueness")
ax[3].set_xlabel("Brightness")
plt.legend()
plt.tight_layout()
fig.savefig(f"post_raster_analysis/brightness_{year}.png")

# %%
precision = results["tp"].shape[1] / (results["tp"].shape[1] + results["fp"].shape[1])
recall = results["tp"].shape[1] / (results["tp"].shape[1] + results["fn"].shape[1])
f1 = 2 * (precision * recall) / (precision + recall)
print(precision, recall, f1)

# %%
ness = {}
for name in results:
    ness[name] = results[name] / results[name].mean(0)

# %%
keep = lambda name: results[name].mean(0) < 140
precision = keep("tp").sum() / (keep("tp").sum() + keep("fp").sum())
recall = keep("tp").sum() / (
    keep("tp").sum() + (~keep("tp")).sum() + ness["fn"].shape[1]
)
f1 = 2 * (precision * recall) / (precision + recall)
print(precision, recall, f1)

# %%
ness["fn"].shape
# %%
plt.scatter(ness["fp"][1], ness["fp"][2])
plt.scatter(ness["tp"][1], ness["tp"][2])


# %%
# Manual inspection
# %%
tiling_path = "./tiling_256_0.25.feather"
labelling_area = gpd.read_file("../labels_rename/selected_area_220404.geojson").to_crs(
    native_crs
)
gdf_tiles = gpd.read_feather(tiling_path).pipe(
    lambda _gdf: _gdf[_gdf.within(labelling_area.unary_union)]
)

for year in (2019, 2021):
    predictions = gpd.read_file(f"results/buffered_polygons_{year}.geojson").to_crs(
        native_crs
    )
    predictions = gpd.overlay(
        predictions, sample, "intersection"
    )  # Select only in sample area
    labels = gpd.read_file(f"../labels_rename/gr_manual_labels_checked_{year}.geojson")
    labels = gpd.GeoDataFrame(
        geometry=list(labels.geometry.buffer(0).unary_union.geoms)
    )  # Fix self-intersection
    predictions = gpd.overlay(predictions, gdf_tiles).set_crs(native_crs)
    labels = gpd.overlay(labels, gdf_tiles).set_crs(native_crs)

    tp = gpd.overlay(predictions[["geometry"]], labels[["geometry"]])
    fp = gpd.overlay(predictions[["geometry"]], labels[["geometry"]], "difference")
    fn = gpd.overlay(labels[["geometry"]], predictions[["geometry"]], "difference")
    fp = fp[fp.area > 10]
    fn = fn[fn.area > 10]
    tp.to_file(f"results/tp_{year}.geojson")
    fp.to_file(f"results/fp_{year}.geojson")
    fn.to_file(f"results/fn_{year}.geojson")
