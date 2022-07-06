import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml

paths = list(Path("./").glob("experiment_*"))
results = []
best_f1 = 0
best_config = ""
# %%
for p in paths:
    config = toml.load(p / "config.toml")
    config["path"] = str(p)
    with open(p / "history.json", "r") as f:
        history = json.load(f)
    # print(config)
    # if config["model_path"]:
    # continue
    # if config["freeze_pretrained"] == 0:
    # continue
    label = f"{config['freeze_pretrained']}; {config['transform']}; {config['lr']:0.1e}"
    for key in history:
        history[key] = np.array(history[key])
    f_score = history["val tp"] / (
        history["val tp"] + 0.5 * (history["val fp"] + history["val fn"])
    )
    f_score_train = history["train tp"] / (
        history["train tp"] + 0.5 * (history["train fp"] + history["train fn"])
    )
    config["f_score"] = f_score[-1]
    config["miou"] = history["val miou"][-1]
    results.append(config)
    y = 1 - f_score
    config["1-f"] = y[-1]
    # y = f_score_train - f_score
    # y = accuracy_n
    plt.plot(y, label=label)

    plt.text(len(y) - 1, y[-1], label)
    plt.yscale("log")
    if f_score[-1] > best_f1:
        best_f1 = f_score[-1]
        best_config = p
        best_config_spec = config
# %%

df = pd.DataFrame(results).sort_values("f_score")
df.to_csv("results.csv")
print(df)

print("Best with no augs")
print(df[df["transform"]=="no_augs"].tail(1).reset_index().to_dict())

print("Best config:", best_config)
print(best_config_spec)
plt.show()
