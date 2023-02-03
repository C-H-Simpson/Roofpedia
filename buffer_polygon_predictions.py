import geopandas as gpd
import pandas as pd
from pathlib import Path

buildings = gpd.read_feather("../data/os_buildings_2021.feather")

for year in (2019, 2021):
    p = Path(f"/home/ucbqc38/Scratch/results/getmapping_{year}_tiled")
    inp = (p / f"merged_polygons_{year}.feather")
    gdf = gpd.read_feather(inp)
    gdf = gpd.overlay(gdf, buildings, "intersection")
    gdf = gpd.GeoDataFrame(geometry=list(gdf.buffer(0).simplify(0.25).unary_union.geoms), crs=gdf.crs)
    gdf = gdf[gdf.area>10]
    outp = p / f"buffered_polygons_{year}.feather"
    gdf.to_feather(outp)
    print(outp)
    outp = p / f"buffered_polygons_{year}.geojson"
    gdf.to_file(outp)
    print(outp)
