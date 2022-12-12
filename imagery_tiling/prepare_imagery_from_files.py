# %%
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import osgeo_utils.gdal_merge
import rasterio
import rasterio.mask
import argparse
from tqdm import tqdm

tqdm.pandas()
gpd.options.use_pygeos = True
from pathlib import Path

if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        help="the directory to which the resulting tiled imagery will be saved",
    )
    parser.add_argument(
        "-g", "--gref10k", help="the OS 10km grid reference that will be processed"
    )
    parser.add_argument(
        "-L",
        "--labels",
        help="the file containing the labelling polygons for the whole region",
    )
    parser.add_argument(
        "-i", "--imagery", help="the directory containing the imagery to be tiled"
    )
    # args = parser.parse_args()
    args = parser.parse_args(
        ["-g", "TQ27", "-i", "/home/ucbqc38/Scratch/getmapping_2021", "-o", "/lustre/scratch/scratch/ucbqc38/getmapping_2021_tiled/TQ27", "-L", "/lustre/home/ucbqc38/Roofpedia_clean/data/gr_manual_labels_221212.geojson"]
    )
    print(args)
    window_height = 256
    window_width = 256
    pixel_size = 0.25

    # %%
    # Load the 1km gridsquares.
    native_crs = "EPSG:27700"
    osgb_1km = gpd.read_file(
        "/home/ucbqc38/Scratch/OSGB_grids/OSGB_Grid_1km.shp"
    ).to_crs(native_crs)

    # %%
    # Load the tiling grid created by "batched_tiling.py"
    gdf_tiles = gpd.read_feather("../data/tiling_256_0.25.feather")
    # Select only those relevant to the grid reference currently being processed.
    gdf_tiles = gdf_tiles[gdf_tiles.TILE_NAME == args.gref10k]

    # %%
    # Create a dict for the input imagery paths to the 1km grid references.
    input_glob = list(Path(args.imagery).glob("*/*/*jpg"))
    input_tiles_path_dict = {g.stem[0:6].upper(): g for g in input_glob}
    assert input_tiles_path_dict

    # %%
    # Buffer the imagery tiles by a pixel width to allow for small overlap errors.
    osgb_1km = osgb_1km.assign(buff=osgb_1km.geometry.buffer(pixel_size))

    # %%
    # Spatially join input and output tiles.
    gdf_links = gdf_tiles[["x", "y", "geometry"]].sjoin(osgb_1km.set_geometry("buff"))[
        ["x", "y", "geometry_left", "PLAN_NO"]
    ]

    # %%
    # Then need to find the minimal input tilessets so we can load each input raster just once
    # something like this
    gdf_tiles = (
        gdf_tiles.set_index(["x", "y"])
        .assign(
            inp_tiles=gdf_links.groupby(["x", "y"]).apply(
                lambda _df: _df.PLAN_NO.sort_values().to_list()
            )
        )
        .reset_index()
    )
    del gdf_links
    gdf_tiles

    # %%
    def query_tile(_df, destination_dir, input_tiles_path_dict):
        """
        For a _df that may contain multiple rows, which are multiple tiles to be clipped out.
        And which might straddle multiple input tiles.
        """
        inp_tiles = _df.inp_tiles.iloc[0]  # should always be the same
        with tempfile.TemporaryDirectory() as tmpdirname:
            if len(inp_tiles) > 1:
                # Merge the input rasters to a temporary file.
                output_path = str(Path(tmpdirname) / "temp.tif")
                input_list = [str(input_tiles_path_dict[t]) for t in inp_tiles if t in input_tiles_path_dict]
                if len(input_list) ==0:
                    print("No imagery tiles", inp_tiles)
                    return False
                parameters = ["", "-o", output_path] + input_list
                osgeo_utils.gdal_merge.main(parameters)
            else:
                t = inp_tiles[0]
                output_path = input_tiles_path_dict[t] if t in input_tiles_path_dict else None
                if output_path is None:
                    print("No imagery tiles", inp_tiles)
                    return False

            # Clip from the temporary file.
            input_path = output_path
            for i, _row in _df.iterrows():
                destination = (
                    Path(destination_dir) / f"{int(_row.x):d}" / f"{int(_row.y):d}.png"
                )
                if destination.is_file():
                    continue

                try:
                    with rasterio.open(input_path) as src:
                        out_image, out_transform = rasterio.mask.mask(
                            src, [_row.geometry], crop=True
                        )
                        out_meta = src.meta
                except ValueError:
                    # If the geometry does not overlap with the raster.
                    print("No imagery", _row)
                    continue

                out_meta.update(
                    {
                        "driver": "PNG",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                    }
                )

                destination.parent.mkdir(exist_ok=True)
                with rasterio.open(destination, "w", **out_meta) as dest:
                    dest.write(out_image)

    # %%
    # Apply the tiling to the whole area.
    # This takes quite a while...
    destination_dir = Path(args.output) / "images"
    destination_dir.mkdir(exist_ok=True)
    print("Splitting imagery")
    gdf_tiles.assign(inp_tiles_str=gdf_tiles.inp_tiles.astype(str)).groupby(
            "inp_tiles_str"
        ).progress_apply(
            lambda _df: query_tile(_df, destination_dir, input_tiles_path_dict)
        )

    # %%
    # Prepare masks from the same tiles.
    from osgeo import ogr, gdal

    shapefile = args.labels

    def write_mask(
        _df,
        window_height,
        window_width,
        pixel_size,
        shapefile,
        destination_dir,
        maskvalue=1,
    ):
        destination = Path(destination_dir) / f"{int(_df.x):d}" / f"{int(_df.y):d}.png"
        if destination.is_file():
            return

        src_ds = ogr.Open(shapefile)
        xmin = _df.x
        ymin = _df.y
        xmax = xmin + window_width
        ymax = ymin + window_height
        ncols = int(window_width / pixel_size)
        nrows = int(window_height / pixel_size)
        xres, yres = pixel_size, pixel_size
        assert xres == (xmax - xmin) / float(ncols)
        assert yres == (ymax - ymin) / float(nrows)
        geotransform = (xmin, xres, 0, ymax, 0, -yres)

        destination.parent.mkdir(exist_ok=True, parents=True)
        destination = destination.resolve().as_posix()

        src_lyr = src_ds.GetLayer()

        dst_ds = gdal.GetDriverByName("MEM").Create("", ncols, nrows, 1, gdal.GDT_Byte)
        dst_rb = dst_ds.GetRasterBand(1)
        dst_rb.Fill(0)  # initialise raster with zeros
        dst_rb.SetNoDataValue(0)
        dst_ds.SetGeoTransform(geotransform)

        err = gdal.RasterizeLayer(dst_ds, [1], src_lyr, burn_values=[maskvalue])

        dst_ds.FlushCache()
        ds2 = gdal.GetDriverByName("PNG").CreateCopy(destination, dst_ds, 0)

        # raise ValueError(destination)

    destination_dir = Path(args.output) / "labels"
    destination_dir.mkdir(exist_ok=True)

    #%%
    print("Making label tiles")
    gdf_tiles.progress_apply(
        lambda _df: write_mask(
            _df,
            window_height,
            window_width,
            pixel_size,
            shapefile,
            destination_dir,
            maskvalue=1,
        ),
        axis=1,
    )
