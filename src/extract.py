import argparse
import os
from pathlib import Path

import osgeo_utils.gdal_merge
import osgeo_utils.gdal_polygonize


def extract(
    input_glob, polygon_output_path, merged_raster_path, nodata=0, format="GeoJSON"
):
    # Merge the predictions
    input_glob = [str(p) for p in input_glob]
    parameters = (
        ["", "-o", merged_raster_path]
        + input_glob
        + [
            "-co",
            "COMPRESS=LZW",
            "-v",
        ]
    )
    if nodata is not None:
        parameters = parameters + ["-n", str(nodata), "-a_nodata", str(nodata)]
    osgeo_utils.gdal_merge.main(parameters)

    # Extract a vector dataset
    parameters = [
        "",
        "-8",
        str(merged_raster_path),
        "-f",
        format,
        str(polygon_output_path),
    ]
    osgeo_utils.gdal_polygonize.main(parameters)
