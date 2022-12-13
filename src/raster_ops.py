"""
Routines to convert prediction images to valid rasters and vectorize.
"""
# %%
from pathlib import Path

import osgeo_utils.gdal_merge
import rasterio

from imagery_tiling.batched_tiling import (
    native_crs,
    pixel_size,
    window_height,
    window_width,
)

# %%
# def tile_to_raster(
#     input_fname: Path, destination_dir: Path, bands: tuple, dtype: str = None
# ):
#     """
#     Turn a mercantile-coded png into a valid geotiff.
#     """
#     dataset = rasterio.open(input_fname, "r")
#     data = dataset.read(bands)
#     crs = {"init": "epsg:4326"}
#     tile = (
#         int(input_fname.parent.stem),
#         int(input_fname.stem),
#         int(input_fname.parent.parent.stem),
#     )
#     bounds = mercantile.bounds(*tile)
#     transform = rasterio.transform.from_bounds(*bounds, data.shape[1], data.shape[2])
#     dest_p = (
#         Path(destination_dir)
#         / input_fname.parent.parent.stem
#         / input_fname.parent.stem
#         / f"{input_fname.stem}.tiff"
#     )
#     dest_p.parent.mkdir(exist_ok=True, parents=True)
#     if dtype is None:
#         dtype = data.dtype
#     with rasterio.open(
#         dest_p,
#         "w",
#         driver="GTiff",
#         width=data.shape[1],
#         height=data.shape[2],
#         count=len(bands),
#         dtype=data.dtype,
#         nodata=0,
#         transform=transform,
#         crs=crs,
#     ) as dst:
#         dst.write(data, indexes=bands)


def tile_to_raster(
    input_fname: Path,
    destination_dir: Path,
    bands: tuple,
    dtype: str = None,
):
    """
    Turn a non-mercantile-coded png into a valid geotiff.
    """
    dataset = rasterio.open(input_fname, "r")
    data = dataset.read(bands)
    crs = {"init": native_crs}  # CRS I assumed in imagery_tiling/batched_tiling.py
    tile = (
        int(input_fname.parent.stem),
        int(input_fname.stem),
        # int(input_fname.parent.parent.stem),
    )
    # bounds = mercantile.bounds(*tile)
    # The bounds assumption I've used in imagery_tiling/batched_tiling.py
    bounds = (
        tile[0],
        tile[1],
        tile[0] + (window_width * pixel_size),
        tile[1] + (window_height * pixel_size),
    )
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


def merge_rasters(input_glob: list, output_path: Path, nodata=0):
    """Use GDAL to merge a list of rasters"""
    # How to merge rasters.
    output_path = str(output_path)
    input_glob = [str(p) for p in input_glob]
    # parameters = ['', '-o', output_path] + input_glob + [ '-co', 'COMPRESS=LZW', "-n", "", "-a_nodata", "0", "-v"]
    parameters = (
        ["", "-o", output_path]
        + input_glob
        + [
            "-co",
            "COMPRESS=LZW",
            "-v",
        ]
    )
    if nodata is not None:
        parameters = parameters + ["-n", str(nodata), "-a_nodata", str(nodata)]
    if Path(output_path).exists():
        raise FileExistsError(output_path)
    osgeo_utils.gdal_merge.main(parameters)


# %%

if __name__ == "__main__":
    # test tile_to_raster
    p = Path(
        r"C:\Users\ucbqc38\Documents\RoofPedia\Roofpedia_vc\dataset\1s\labels\19\262051\174218.png"
    )
    tile_to_raster(p, Path("./test_rasterops"), (1,), "int8")

    # %%
    # Test tile_to_raster on a larger scale
    input = Path(
        r"C:\Users\ucbqc38\Documents\RoofPedia\Roofpedia_vc\dataset\1s\labels\19"
    ).glob("*/*png")
    input = list(input)
    [tile_to_raster(p, Path("./test_rasterops"), (1,), "int8") for p in input]

    # %%
    output_file_path = r"test_rasterops/merged.tiff"
    input_files_path = [str(p) for p in Path(r"test_rasterops").glob("*/*/*.tiff")]
    merge_rasters(input_files_path, output_file_path)

    # %%
    # How to vectorize rasters.
    import osgeo_utils.gdal_polygonize

    parameters = [
        "",
        "-8",
        "" "test_rasterops/merged.tiff",
        "-f",
        "GeoJSON",
        "./test_rasterops/merged.geojson",
    ]
    osgeo_utils.gdal_polygonize.main(parameters)
