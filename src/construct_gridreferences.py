"""
Divide London into gridsquares for paralell prediction.
"""
from os import symlink
import geopandas as gpd
from pathlib import Path
import mercantile
import shapely

gdf = gpd.read_file("../data/OSGB_10km_shp/OSGB_Grid_10km.shp")
gdf_london = gpd.read_file("../data/London_borough_shp/London_Borough_Excluding_MHW.shp")
gdf = gpd.overlay(gdf, gdf_london)
gdf=gdf.to_crs("EPSG:4326")

ftiles = list(Path("results/02Images/GreaterLondon").glob("*/*/*png"))
tiles = [
    (int(Path(s).parent.stem), int(Path(s).stem), int(Path(s).parent.parent.stem))
    for s in ftiles
]
bounds = [mercantile.bounds(*s) for s in tiles]
boxes = [shapely.geometry.box(*b) for b in bounds]
gdf_tiles = gpd.GeoDataFrame(
    {"label_tiles": [str(a) for a in ftiles]}, geometry=boxes, crs="EPSG:4326"
)

# %%
gdf_tiles = gdf_tiles.sjoin(gdf[["geometry", "TILE_NAME"]])
print(gdf_tiles)

# %%
for TILE_NAME, df in gdf_tiles[["label_tiles", "TILE_NAME"]].groupby("TILE_NAME"):
    destination_dir = Path(f"results/02Images/{TILE_NAME}")
    paths = [Path(p) for p in df.label_tiles]
    for p in df.label_tiles:
        p = Path(p).resolve()
        dest = destination_dir / p.parent.parent.stem / p.parent.stem / p.name
        dest.parent.mkdir(exist_ok=True, parents=True)
        symlink(p, dest)

# %%
# Save the list of grid references
with open("grid_references.txt", 'w') as f:
    for gref in gdf.TILE_NAME.values:
       f.write(f"{gref}\n") 
    
