# %%
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import osgeo_utils.gdal_merge
import pygeos
import rasterio
import rasterio.mask
from tqdm import tqdm

tqdm.pandas()
from pathlib import Path

# %%
if __name__ == "__main__":
    pass
    # %%
    # The first technique I tried wasn't really practical.
    # 
    # # First the imagery has to be merged to a single large file.
    # # This takes about 12 minutes for a 10km2 tile, so about 3 hours for all
    # # of Greater London.
    # # The file size is about 4GB for 10km2, so about 600GB for the whole of London.
    # # CF about 40 GB which is the original imagery.
    # output_path = "./test_imagery_merge.tiff"
    # input_glob = [
    #     str(p)
    #     for p in (
    #         Path(
    #             r"C:\Users\ucbqc38\Documents\GIS\getmapping_latest_london_imagery\Download_tq38_1829916\getmapping_rgb_25cm_4194129\tq"
    #         ).glob("*jpg")
    #     )
    # ]
    # # parameters = ['', '-o', output_path] + input_glob + [ '-co', 'COMPRESS=LZW', "-n", "", "-a_nodata", "0", "-v"]
    # parameters = (
    #     ["", "-o", output_path]
    #     + input_glob
    #     + [
    #         "-co",
    #         "COMPRESS=JPEG",
    #         "-v",
    #     ]
    # )
    # if Path(output_path).exists():
    #     raise FileExistsError(output_path)
    # osgeo_utils.gdal_merge.main(parameters)

    # # %%
    # # Then it can be tiled.
    # input = output_path
    # output = Path("./tiling_test")
    # parameters = [
    #     "--profile=mercator",
    #     "-r",
    #     "near",
    #     "-s",
    #     "EPSG:27700",
    #     "--xyz",
    #     "-z",
    #     "19",
    #     "--tilesize",
    #     "256",
    #     "--processes",
    #     "8",
    #     "-v",
    #     str(input),
    #     str(output),
    # ]
    # print(parameters)
    # osgeo_utils.gdal2tiles.main(parameters)


    # %%
    # Load the 1km gridsquares.
    gpd.options.use_pygeos = True
    native_crs = "EPSG:27700"
    osgb_1km = gpd.read_file("../../OSGB_Grids/Shapefile/OSGB_Grid_1km.shp").to_crs(
        native_crs
    )
    
    # %%
    # Get boundaries for London
    gdf_london = gpd.read_file("../../GIS/statistical-gis-boundaries-london/ESRI/London_Ward.shp").to_crs(native_crs)
    london = gdf_london.dissolve().geometry.item()

    # %%
    # Specify the properties of the tiling.
    pitch = 256
    pixel_size = 0.25
    window_width = pitch * pixel_size # Ideally a whole number of metres
    window_height = window_width
    domain_west = 530_000
    domain_south = 181_000
    domain_east = 536_000 + window_width
    domain_north = 187_000 + window_width

    # %%
    # OR, specify bounds from London geometry
    domain_west, domain_south, domain_east, domain_north = gdf_london.total_bounds.round(0)
    domain_west = domain_west  - window_width
    domain_south = domain_south  - window_height
    domain_east = domain_east + window_width
    domain_north = domain_north + window_height
    domain_west, domain_south, domain_east, domain_north 

    # %%
    # Create a dict for the input imagery paths to the 1km grid references.
    input_glob = list(
        Path(r"C:\Users\ucbqc38\Documents\GIS\getmapping_latest_london_imagery").glob(
            "*/*/*/*jpg"
        )
    )
    input_tiles_path_dict = {g.stem[0:6].upper(): g for g in input_glob}

    # %%
    # Use pygeos to construct tiles
    xy_array = np.mgrid[domain_west:domain_east:window_width, domain_south:domain_north:window_height].T.reshape(-1,2)
    boxes = pygeos.box(xy_array[:,0], xy_array[:,1], xy_array[:,0]+window_width, xy_array[:,1]+window_height)
    gdf_tiles = gpd.GeoDataFrame(xy_array, geometry=boxes, crs=native_crs).rename(columns={0: "x", 1: "y"})
    del boxes, xy_array

    # %%
    # Make sure the tiles are all in London.
    # The tile construction routine will initially create some tiles outside London as the bounds are rectangular.
    print(len(gdf_tiles), "tiles before intersection with london")
    gdf_tiles = gdf_tiles.iloc[gdf_tiles.sindex.query(london, "intersects")]
    print(len(gdf_tiles), "tiles after intersection with london")

    # %%
    # Buffer the imagery tiles by a pixel width to allow for small overlap errors.
    osgb_1km = osgb_1km.assign(buff=osgb_1km.geometry.buffer(pixel_size))
    osgb_1km = osgb_1km[osgb_1km.intersects(london)]

    # %%
    # Spatially join input and output tiles.
    gdf_links = gdf_tiles.sjoin(osgb_1km.set_geometry("buff"))[["x", "y", "geometry_left", "PLAN_NO"]]

    # %%
    # Then need to find the minimal input tilessets so we can load each input raster just once
    # something like this
    gdf_tiles = gdf_tiles.set_index(["x", "y"]).assign(inp_tiles=gdf_links.groupby(["x", "y"]).apply(lambda _df: _df.PLAN_NO.sort_values().to_list())).reset_index()
    del gdf_links
    gdf_tiles

    # %%
    def query_tile(_df, destination_dir, input_tiles_path_dict):
        """
        For a _df that may contain multiple rows, which are multiple tiles to be clipped out.
        And which might straddle multiple input tiles.
        """
        inp_tiles = _df.inp_tiles.iloc[0] # should always be the same
        with tempfile.TemporaryDirectory() as tmpdirname:
            if len(inp_tiles) > 1:
            # Merge the input rasters to a temporary file.
                output_path = str(Path(tmpdirname) / "temp.tif")
                input_list = [ str(input_tiles_path_dict[t]) for t in inp_tiles]
                parameters = (
                    ["", "-o", output_path]
                    + input_list
                )
                osgeo_utils.gdal_merge.main(parameters)
            else:
                output_path = input_tiles_path_dict[inp_tiles[0]]

            # Clip from the temporary file.
            input_path = output_path
            for i, _row in _df.iterrows():
                with rasterio.open(input_path) as src:
                    out_image, out_transform = rasterio.mask.mask(src, [_row.geometry], crop=True)
                    out_meta = src.meta

                out_meta.update({"driver": "PNG",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform})

                destination = Path(destination_dir) / f"{int(_row.x):d}" / f"{int(_row.y):d}.png"
                destination.parent.mkdir(exist_ok=True)
                with rasterio.open(destination, "w", **out_meta) as dest:
                    dest.write(out_image)

        return ""


    # %%
    # Apply the tiling to the whole area.
    # This takes quite a while...
    destination_dir = Path("./test_manual_tiling")
    destination_dir.mkdir(exist_ok=True)
    gdf_tiles.assign(inp_tiles_str=gdf_tiles.inp_tiles.astype(str)).groupby("inp_tiles_str").progress_apply(lambda _df: query_tile(_df, destination_dir, input_tiles_path_dict))

# %%
