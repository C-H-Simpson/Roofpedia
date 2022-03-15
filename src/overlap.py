"""
The registration between the building footprints and the imagery is imperfect.
Therefore, it makes sense to be a bit more relaxed about the intersection of our
identified green roof areas and building footprints.

In this routine, we take the overlay intersection of the green roofs and  the
building footprints, and set a limit on the fraction of the green roof  that
intersects.

This threshold is a matter for optimization. In my own work, I find that an
overlap threshold of 0.8 and area threshold of 0 works well. Your results will
vary on the source of imagery and building footprints.

The area threshold might not be very important if you have already done
smoothing or binary opening of the results.
"""

import geopandas as gpd
import argparse


def overlapping_threshold(gdf_bui, gdf_feat, min_area, min_overlap):
    """Keep only features overlapping with building footprints.

    gdf_bui: a GeoDataFrame of building footprints as polygons or multipolygons.
    gdf_feat: a GeoDataFrame of feature patches e.g. green roof identified from
        imagery.
    min_area: float, minimum area of feature patches. Set to 0 to accept all.
    min_overlap: float, minimum fraction of feature patch that must overlap
        with the building footprints in order for the patch to be kept.

    returns:
        a GeoDataFrame of patches that meet the requirements.
    """
    if gdf_bui.crs != gdf_feat.crs:
        raise ValueError("GeoDataFrames must have same CRS.")
    if min_area < 0 or not isinstance(min_area, float):
        raise ValueError("min_area must be a float >= 0.")
    if min_overlap < 0 or min_overlap > 1 or not isinstance(min_overlap, float):
        raise ValueError("min_overlap must be a float between 0 and 1")

    gdf_feat["feat_idx"] = gdf_feat.index
    gdf_bui["bui_idx"] = gdf_bui.index
    gdf_feat["cov_area"] = gdf_feat.area
    gdf_feat_contained = gpd.overlay(
        gdf_feat[["geometry", "feat_idx", "cov_area"]],
        gdf_bui[["geometry", "bui_idx"]],
        how="intersection",
    )
    # Get the fraction of the feature that overlaps with the building footprint.
    gdf_feat_contained["prop_cont"] = (
        gdf_feat_contained.area / gdf_feat_contained.cov_area
    )
    gdf_feat_contained = gdf_feat_contained[
        (gdf_feat_contained.area > min_area)
        & (gdf_feat_contained.prop_cont > min_overlap)
    ]
    if gdf_feat_contained.empty:
        result = None
    else:
        result = gdf_feat.set_index("feat_idx").loc[gdf_feat_contained.feat_idx]
    return gdf_feat_contained


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "city", help="City to be predicted, must be the same as the name of the dataset"
    )
    parser.add_argument(
        "type", help="Roof Typology, Green for Greenroof, Solar for PV Roof"
    )
    parser.add_argument("crs", help="CRS in which to perform the geometric operations.")
    parser.add_argument(
        "min_area", help="Minimum area for a feature polygon.", type=float
    )
    parser.add_argument(
        "min_overlap",
        help="Minimum overlap between feature an and building footprints.",
        type=float,
    )
    args = parser.parse_args()

    city_name = args.city
    target_type = args.type

    feature_path = "results/04Results/" + city_name + "_" + target_type + "_raw.geojson"
    city_path = "results/01City/" + city_name + ".geojson"

    gdf_city = gpd.read_file(city_path)[["geometry"]].to_crs(args.crs)
    gdf_features = gpd.read_file(feature_path)[["geometry"]].to_crs(args.crs)
    gdf_result = overlapping_threshold(
        gdf_city, gdf_features, args.min_area, args.min_overlap
    )
    gdf_result.to_file(
        "results/04Results/" + city_name + "_" + target_type + "_ot.geojson",
        driver="GeoJSON",
    )
