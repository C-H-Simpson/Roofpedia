"""
Routines to convert prediction images to valid rasters and vectorize.
"""

# %%
import rasterio
from pathlib import Path
import mercantile

# %%

# %%
def tile_to_raster(input_fname: Path, destination_dir: Path, bands: tuple, dtype: str=None):
    """
    Turn a mercantile-coded png into a valid geotiff.
    """
    dataset = rasterio.open(input_fname, "r")
    data = dataset.read(bands)
    crs = {"init": "epsg:4326"}
    tile = (
        int(input_fname.parent.stem),
        int(input_fname.stem),
        int(input_fname.parent.parent.stem),
    )
    bounds = mercantile.bounds(*tile)
    transform = rasterio.transform.from_bounds(*bounds, data.shape[1], data.shape[2])
    dest_p = (
        Path(destination_dir)
        / input_fname.parent.parent.stem
        / input_fname.parent.stem
        / f"{input_fname.stem}.tiff"
    )
    dest_p.parent.mkdir(exist_ok=True, parents=True)
    if dtype is None:
        dtype = data.dtype
    with rasterio.open(
        dest_p,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[2],
        count=len(bands),
        dtype=data.dtype,
        nodata=0,
        transform=transform,
        crs=crs,
    ) as dst:
        dst.write(data, indexes=bands)

# %%
# test tile_to_raster
p = Path(
    r"C:\Users\ucbqc38\Documents\RoofPedia\Roofpedia_vc\dataset\1s\labels\19\262051\174218.png"
)
tile_to_raster(p, Path("./test_rasterops"), (1,), "int8")

# %%
# Test tile_to_raster on a larger scale
input = Path(
    r"C:\Users\ucbqc38\Documents\RoofPedia\Roofpedia_vc\dataset\1s\labels\19").glob("*/*png")
input = list(input)
[tile_to_raster(p, Path("./test_rasterops"), (1,), "int16") for p in input]


# %%
# How to merge rasters.
import osgeo_utils.gdal_merge
output_file_path = r'test_rasterops/merged.tiff'
input_files_path = [str(p) for p in Path(r'test_rasterops').glob('*/*/*.tiff')]
parameters = ['', '-o', output_file_path] + input_files_path + [ '-co', 'COMPRESS=LZW', "-n", "0", "-a_nodata", "0", "-v"]
osgeo_utils.gdal_merge.main(parameters)

# %%
# How to vectorize rasters.
import osgeo_utils.gdal_polygonize
parameters = ["",
    # "-8",
    "test_rasterops/merged.tiff", "-f", "GeoJSON", "./test_rasterops/merged.geojson"]
osgeo_utils.gdal_polygonize.main(parameters)