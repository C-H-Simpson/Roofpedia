from pathlib import Path
import shapely.geometry
import geopandas as gpd
import mercantile

files = {}
tiles = {}
bounds= {}
boxes = {}
for ds in ("training", "validation", "evaluation"):
    files[ds] = list(Path(f"dataset/{ds}/labels/19").glob("*/*.png"))
    tiles[ds] = [(int(s.parent.stem), int(s.stem), int(s.parent.parent.stem)) for s in files[ds]]
    bounds[ds] = [mercantile.bounds(*s) for s in tiles[ds]]
    boxes[ds] = [shapely.geometry.box(*b) for b in bounds[ds]]
    gdf = gpd.GeoDataFrame(geometry=boxes[ds])
    gdf["ds"] = ds
    gdf.to_file(f"dataset/{ds}.geojson", driver="GeoJSON")
    


