"""
Find the raw imagery
Slice up this area into 10km grid references
Write bash scripts that slice up each 10km grid reference into 256 pixel imagery
tiles using prepare_imagery_from_files.py
Submit the bash scripts to the myriad queue
"""
# %%
import subprocess
from pathlib import Path

import geopandas as gpd
import numpy as np
import pygeos

gpd.options.use_pygeos = True

# %%
# Specify the properties of the tiling.
pitch = 256
pixel_size = 0.25
window_width = pitch * pixel_size  # Ideally a whole number of metres
window_height = window_width

# %%
# specify bounds from London geometry
native_crs = "EPSG:27700"
gdf_london = gpd.read_file(
    "../data/London_borough_shp/London_Borough_Excluding_MHW.shp"
).to_crs(native_crs)
domain_west, domain_south, domain_east, domain_north = gdf_london.total_bounds.round(0)
domain_west = domain_west - window_width
domain_south = domain_south - window_height
domain_east = domain_east + window_width
domain_north = domain_north + window_height
domain_west, domain_south, domain_east, domain_north

label_polygon_path = {
    "getmapping_2019": Path("../data/gr_manual_labels_checked_2019.geojson").resolve(),
    "getmapping_2021": Path("../data/gr_manual_labels_checked_2021.geojson").resolve()
}

# A geodataframe will be created in this location which identifies the tiles.
tiling_path = f"../data/tiling_{pitch}_{pixel_size}.feather"

# %%
if __name__ == "__main__":
    # %%
    # Find the raw imagery.
    imagery_2019_glob = list(
        Path("/home/ucbqc38/Scratch/getmapping_2019").glob("*/*/*jpg")
    )
    assert len(imagery_2019_glob)

    # %%
    imagery_2021_glob = list(
        Path("/home/ucbqc38/Scratch/getmapping_2021").glob("*/*/*jpg")
    )
    assert len(imagery_2021_glob)

    # %%
    # Create a dict for the input imagery paths to the 1km grid references.
    imagery_2019_dict = {g.stem[0:6].upper(): g for g in imagery_2019_glob}
    imagery_2021_dict = {g.stem[0:6].upper(): g for g in imagery_2021_glob}

    # %%
    # Get 10km grid references.
    osgb_10km = gpd.read_file(
        "/home/ucbqc38/Scratch/OSGB_grids/OSGB_Grid_10km.shp"
    ).to_crs(native_crs)
    london = gdf_london.dissolve().geometry.item()
    gdf = gpd.overlay(osgb_10km, gdf_london.dissolve())
    tile_names_10km = gdf.TILE_NAME.to_list()

    # %%
    # The raw imagery is in 1km tiles to begin with.
    # Get 1km grid references.
    gdf = gpd.read_file("/home/ucbqc38/Scratch/OSGB_grids/OSGB_Grid_1km.shp").to_crs(
        native_crs
    )
    gdf = gpd.overlay(gdf.to_crs(native_crs), gdf_london.to_crs(native_crs).dissolve())
    tile_names_1km = gdf.PLAN_NO.to_list()

    # %%
    # We need to construct the "grid" of small tiles into which we will slice
    # our imagery.

    # %%
    # Use pygeos to construct tiles
    xy_array = np.mgrid[
        domain_west:domain_east:window_width, domain_south:domain_north:window_height
    ].T.reshape(-1, 2)
    boxes = pygeos.box(
        xy_array[:, 0],
        xy_array[:, 1],
        xy_array[:, 0] + window_width,
        xy_array[:, 1] + window_height,
    )
    gdf_tiles = (
        gpd.GeoDataFrame(xy_array, geometry=boxes, crs=native_crs)
        .rename(columns={0: "x", 1: "y"})
        .set_crs(native_crs)
    )
    gdf_tiles = gdf_tiles.iloc[gdf_tiles.sindex.query(london, "intersects")]
    del boxes, xy_array

    # %%
    gdf_tiles["vertex"] = pygeos.points(gdf_tiles.x, gdf_tiles.y)
    # %%
    # Assign each of these small tiles to a 10km gridcell
    gdf_tiles = (
        gdf_tiles.set_geometry("vertex")
        .set_crs(native_crs)
        .sjoin(osgb_10km[["geometry", "TILE_NAME"]])
        .set_geometry("geometry")
    )

    # %%
    # Save the grid
    gdf_tiles.to_feather(tiling_path)

    # %%
    # Generate a version of the script with the right input, then submit to queue.
    script = Path("imagery_tiling/batched_tiling.sh").read_text()
    for dset in ("getmapping_2021", "getmapping_2019"):
        destination_dir = Path("/home/ucbqc38/Scratch") / f"{dset}_tiled"
        imagery_dir = f"/home/ucbqc38/Scratch/{dset}"
        if destination_dir.is_dir():
            raise FileExistsError(str(destination_dir))
        destination_dir.mkdir()

        for gref10k in tile_names_10km:
            destination = destination_dir / gref10k
            if destination.is_dir():
                raise FileExistsError(str(destination))
            destination.mkdir()

            e_path = str((destination / "create.e").resolve())
            o_path = str((destination / "create.o").resolve())

            script_local = (
                script.replace("$gref10k", gref10k)
                .replace("$destination", str(destination.resolve()))
                .replace(
                    "$labels",
                    str(label_polygon_path[dset]),
                )
                .replace("$ERRFILE", e_path)
                .replace("$OUTFILE", o_path)
                .replace("$imagery_dir", imagery_dir)
            )
            script_path = destination / "create.sh"
            script_path.write_text(script_local)

            print(script_path, o_path)

            # Submit the script
            print(subprocess.check_output(["qsub", str(script_path.resolve())]))

    # %%
