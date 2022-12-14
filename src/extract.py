import tempfile
from pathlib import Path

import geopandas as gpd
import osgeo_utils.gdal_merge
import osgeo_utils.gdal_polygonize
import pandas as pd


def extract(
    input_glob, polygon_output_path, format="GeoJSON", force_crs="EPSG:27700"
):
    input_glob = list(input_glob)
    with tempfile.TemporaryDirectory() as tmpd:
        polygon_temp_paths = [Path(tmpd) / (p.name+".geojson") for p in input_glob]
        for p, p_poly in zip(input_glob, polygon_temp_paths):
            p = str(p)
            # Extract a vector dataset
            parameters = [
                "",
                "-8",
                str(p),
                "-f",
                format,
                str(p_poly),
            ]
            osgeo_utils.gdal_polygonize.main(parameters)

        pd.concat( (gpd.read_file(p)for p in polygon_temp_paths)) .set_crs(force_crs, allow_override=True).to_file(polygon_output_path)
