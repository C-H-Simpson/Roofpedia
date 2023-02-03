# %%
import geopandas as gpd
from pathlib import Path

crs = "EPSG:27700"

# %%

# %%
gdf_boroughs = gpd.read_file(
    Path(
        "../../greenroofs_analysis/data/Boroughs/ESRI/London_Borough_Excluding_MHW.shp"
    )
).to_crs(crs)
gdf_boroughs = gdf_boroughs.rename(columns={"NAME": "LAD11NM"})
gdf_inner = gdf_boroughs[gdf_boroughs.ONS_INNER == "T"]
gdf_domain = gdf_boroughs
# gdf_domain = gdf_inner
gdf_caz = gpd.read_file(
    "../../greenroofs_analysis/data/CAZ_boundary/lp-consultation-oct-2009-central-activities-zone.shp"
).to_crs(crs)
# %%
gdf = gpd.read_file("../labels_rename/gr_manual_labels_checked_2019.geojson")
print(
    gpd.overlay(gdf, gdf_caz).area.sum(),
    gpd.overlay(gdf, gdf_caz, "difference").area.sum(),
)
# %%
gdf = gpd.read_file("../labels_rename/gr_manual_labels_checked_2021.geojson")
print(
    gpd.overlay(gdf, gdf_caz).area.sum(),
    gpd.overlay(gdf, gdf_caz, "difference").area.sum(),
)
# %%
