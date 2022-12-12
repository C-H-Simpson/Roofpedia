# %%
from osgeo import gdal, ogr
import geopandas as gpd

# %%
native_crs = "EPSG:27700"
gdf_london = gpd.read_file("../../GIS/statistical-gis-boundaries-london/ESRI/London_Ward.shp").to_crs(native_crs)
# specify bounds from London geometry
# Must be the same method as in prepare_imagery_from_files.py
pitch = 256
pixel_size = 0.25
window_width = pitch * pixel_size # Ideally a whole number of metres
window_height = window_width
domain_west, domain_south, domain_east, domain_north = gdf_london.total_bounds.round(0)
domain_west = domain_west  - window_width
domain_south = domain_south  - window_height
domain_east = domain_east + window_width
domain_north = domain_north + window_height
domain_west, domain_south, domain_east, domain_north 
# # %%
# # Define NoData value of new raster
# NoData_value = 0

# # Filename of input OGR file
# vector_fn = r"C:\Users\ucbqc38\Documents\RoofPedia\gr_manual_labels_20220401.geojson"

# # Filename of the raster Tiff that will be created
# raster_fn = "gr_manual_labels_raster.tif"

# # Open the data source and read in the extent
# source_ds = gdal.OpenEx(vector_fn)
# pixel_size = 2

# gdal.Rasterize(raster_fn,
#     vector_fn, format='GTIFF', outputType=gdal.GDT_Byte,
#     creationOptions=["COMPRESS=LZW"], noData=NoData_value,
#     initValues=NoData_value, xRes=pixel_size, yRes=-pixel_size, allTouched=True, burnValues=255,
#     # outputBounds=[domain_west, domain_south, domain_east, domain_north])
# )

# %%
# Rasterization recipe.
shapefile = r"C:\Users\ucbqc38\Documents\RoofPedia\gr_manual_labels_20220401.geojson"
src_ds = ogr.Open(shapefile)
xmin,ymin,xmax,ymax=domain_west, domain_south, domain_east, domain_north

def get_mask_from_poly(ncols, nrows, xmin, ymin, xmax, ymax, src_ds, maskvalue=1):
    assert xres==(xmax-xmin)/float(ncols)
    assert yres==(ymax-ymin)/float(nrows)
    geotransform=(xmin,xres,0,ymax,0, -yres)

    src_lyr=src_ds.GetLayer()

    dst_ds = gdal.GetDriverByName('MEM').Create('', ncols, nrows, 1 ,gdal.GDT_Byte)
    dst_rb = dst_ds.GetRasterBand(1)
    dst_rb.Fill(0) #initialise raster with zeros
    dst_rb.SetNoDataValue(0)
    dst_ds.SetGeoTransform(geotransform)

    err = gdal.RasterizeLayer(dst_ds, [1], src_lyr, burn_values=[maskvalue])

    dst_ds.FlushCache()

    mask_arr=dst_ds.GetRasterBand(1).ReadAsArray()

    return mask_arr

# %%
mask_arr.any()

# %%
