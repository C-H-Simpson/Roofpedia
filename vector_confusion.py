# %%
import geopandas as gpd

# %%
predictions = gpd.read_file(
    r"C:\Users\ucbqc38\Documents\greenroofs_analysis\data\predictions_intersected_nomorph_220719\postprocess_merge_cut.geojson"
)
truth = gpd.read_file(r"C:\Users\ucbqc38\Documents\gr_manual_labels_20220128.gpkg")
selected_area = gpd.read_file(
    r"C:\Users\ucbqc38\Documents\RoofPedia\data_220401\selected_area_220404.gpkg"
)

# %%
predictions = gpd.overlay(predictions, selected_area, "intersection")

# %%
fp = gpd.overlay(predictions, truth, "difference")
fp.to_file("fp.geojson")
# %%
fn = gpd.overlay(truth, predictions, "difference")
fn.to_file("fn.geojson")
# %%
