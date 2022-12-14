# %%
from pathlib import Path
# from imagery_tiling.batched_tiling import tiling_path
# import geopandas as gpd

# %%
pixel_size = 0.25
window_size = 256 * pixel_size
gi = list(Path("dataset").glob("k*/*/images/*/*png"))
assert gi
xy = ((p, float(p.parent.stem), float(p.stem)+window_size) for p in gi)
meta = (
    (
        Path(str(p)+".aux.xml"),
        f"""<PAMDataset>
<GeoTransform>  {xmin:e},  {pixel_size:e},  {0:e},  {ymin:e},  {0:e}, -{pixel_size:e}</GeoTransform>
<Metadata domain="IMAGE_STRUCTURE">
    <MDI key="INTERLEAVE">PIXEL</MDI>
</Metadata>
</PAMDataset>
"""
    )
    for p, xmin, ymin in xy
)
for p, m in meta:
    p.write_text(m)

# %%
