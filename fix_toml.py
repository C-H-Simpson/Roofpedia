# %%
from pathlib import Path

# %%
for p in Path("results").glob("*/*toml"):
    f = p.read_text()
    f = f.replace("[ 1, 1.0,]", "[ 1.0, 1.0,]")
    p.write_text(f)
    f = p.read_text()
    print(f)
# %%
f

# %%
p

# %%
list(Path("results").glob("*/*toml"))
# %%
