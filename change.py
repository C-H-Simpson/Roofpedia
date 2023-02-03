# %%
import geopandas as gpd
from pathlib import Path

crs = "EPSG:27700"

# %%
data = {year: gpd.read_file(f"results/buffered_polygons_{year}.geojson") for year in (2021, 2019)}
# %%
change = gpd.overlay(
    data[2021],
    data[2019].assign(geometry=data[2019].buffer(5)),
    "difference"
)

# %%
change = change.assign(geometry=change.geometry.simplify(tolerance=0.25))
# %%
change = change[change.area>10]
# %%
change.to_file("results/change_2019_2021.geojson", index=False)

# %%
change.area.sum()/1e4

# %%
gdf_boroughs = gpd.read_file(
    Path("../../greenroofs_analysis/data/Boroughs/ESRI/London_Borough_Excluding_MHW.shp")
).to_crs(crs)
gdf_boroughs = gdf_boroughs.rename(columns={"NAME": "LAD11NM"})
gdf_inner = gdf_boroughs[gdf_boroughs.ONS_INNER == "T"]
gdf_domain = gdf_boroughs
# gdf_domain = gdf_inner
gdf_caz = gpd.read_file(
    "../../greenroofs_analysis/data/CAZ_boundary/lp-consultation-oct-2009-central-activities-zone.shp"
).to_crs(crs)


# %%
gpd.overlay(change, gdf_caz).area.sum() / 1e4

# %%
gpd.overlay(change, gdf_inner).area.sum() / 1e4