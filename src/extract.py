# %%
from pathlib import Path

import geopandas as gpd
#import osgeo_utils.gdal_merge
#import osgeo_utils.gdal_polygonize
import pandas as pd

import rasterio
from rasterio import features
import shapely
from tqdm import tqdm

def extract(input_glob, polygon_output_path, force_crs="EPSG:27700"):
    input_glob = list(input_glob)
    polygon_temp_paths = [p.parent / (p.name + ".geojson") for p in input_glob]
    valid_polygon_paths = []
    for p, p_poly in zip(input_glob, tqdm(polygon_temp_paths, ascii=True)):
        p = str(p)
        with rasterio.open(p, "r") as src:
            mask = src.read(1)
            transform = src.transform
        shapes = features.shapes(mask, mask=mask, transform=transform)
        gdf = gpd.GeoDataFrame(geometry=[shapely.geometry.shape(s) for s,v in shapes], crs=force_crs)
        if not gdf.empty:
            gdf.to_file(p_poly, index=False)
            valid_polygon_paths.append(p_poly)

    if valid_polygon_paths:
        gdf = pd.concat((gpd.read_file(p) for p in valid_polygon_paths)).set_crs(
            force_crs, allow_override=True
        )
        gdf = gpd.GeoDataFrame(geometry=list(gdf.unary_union.geoms), crs=force_crs)
        gdf.to_file(polygon_output_path)
    else:
        raise ValueError("No polygons found")
