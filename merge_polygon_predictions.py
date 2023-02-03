import geopandas as gpd
import pandas as pd
from pathlib import Path

for year in (2019, 2021):
    p = Path(f"/home/ucbqc38/Scratch/results/getmapping_{year}_tiled")
    g = list(p.glob(f"*/getmapping_{year}_tiled.geojson"))
    assert g
    gdf = pd.concat((gpd.read_file(f) for f in g))
    #gdf.to_file(p / f"merged_polygons_{year}.geojson", driver="GeoJSON")
    gdf.to_feather(p / f"merged_polygons_{year}.feather")
