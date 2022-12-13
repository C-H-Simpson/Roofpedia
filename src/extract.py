import argparse
import os

from pathlib import Path

import osgeo_utils.gdal_polygonize
import osgeo_utils.gdal_merge


def extract(
    input_glob, polygon_output_path, merged_raster_path, nodata=0, format="GeoJSON"
):
    # Merge the predictions
    assert type(input_glob) == list
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "city", help="City to be predicted, must be the same as the name of the dataset"
    )
    parser.add_argument(
        "type", help="Roof Typology, Green for Greenroof, Solar for PV Roof"
    )
    args = parser.parse_args()

    city_name = args.city
    target_type = args.type
    mask_dir = os.path.join("results", "03Masks", target_type, city_name)

    format = "GeoJSON"
    polygon_output_path = Path("results") / args.city_name / "polygons.geojson"
    merged_raster_path = Path("results") / args.city_name / "merged.tif"

    mask_glob = list((Path("results") / args.city_name / "predictions").glob("*/*png"))

    extract(
        mask_glob=mask_glob,
        polygon_output_path=polygon_output_path,
        merged_raster_path=merged_raster_path,
        format=format,
    )
