# %%
import json
import shutil
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml

matplotlib.use("TKAgg")

paths = list(Path("results").glob("experiment_*"))
assert len(paths) > 0
results = []
best_f1 = 0
best_config = ""
# %%
for p in paths:
    try:
        config = toml.load(p / "config.toml")
    except toml.decoder.TomlDecodeError:
        print("Could not parse", p)
        continue
    config["path"] = str(p)
    history_p = p / "history.json"
    if not (history_p).is_file():
        print("Could not find", history_p)
        continue
    with open(history_p, "r") as f:
        history = json.load(f)
    # print(config)
    # if config["model_path"]:
    # continue
    # if config["freeze_pretrained"] == 0:
    # continue
    for key in history:
        history[key] = np.array(history[key])
    f_score = history["val tp"] / (
        history["val tp"] + 0.5 * (history["val fp"] + history["val fn"])
    )
    f_score_train = history["train tp"] / (
        history["train tp"] + 0.5 * (history["train fp"] + history["train fn"])
    )
    config["accuracy_train"] = (
        (history["train tp"] + history["train tn"])
        / (
            history["train tp"]
            + history["train tn"]
            + history["train fp"]
            + history["train fn"]
        )
    )[-1]
    config["accuracy_val"] = (
        (history["val tp"] + history["val tn"])
        / (
            history["val tp"]
            + history["val tn"]
            + history["val fp"]
            + history["val fn"]
        )
    )[-1]

    config["f_score"] = f_score[-1]
    config["miou"] = history["val miou"][-1]

    config["precision"] = history["val tp"][-1] / (
        history["val tp"][-1] + history["val fp"][-1]
    )
    config["recall"] = history["val tp"][-1] / (
        history["val tp"][-1] + history["val fn"][-1]
    )

    # y = f_score_train - f_score
    # y = accuracy_n
    y = 1 - f_score
    plt.plot(1 - f_score)

    plt.yscale("log")

    if "checkpoint_path" not in config:
        print("No checkpoint in config")
    elif not (Path(config["checkpoint_path"]) / "final_checkpoint.pth").exists():
        print(f"Checkpoint wasn't saved correctly for {p}")
    else:
        print(f"Saving results of {p}")
        results.append(config)

        if f_score[-1] > best_f1:
            best_f1 = f_score[-1]
            best_config = p
            best_config_spec = config

print(f"{best_f1}, {best_config}")
print(best_config_spec)

df = pd.DataFrame(results)
print(df.head())
df = df.sort_values("f_score")
df.to_csv("results.csv")
print(df)

# Get best for a given set of parameters (across learning rates).
df_best_lr = (
    df.groupby(["loss_func", "transform_name", "freeze_pretrained"])
    .apply(lambda _df: _df.iloc[_df.f_score.argmax()])
    .sort_values("f_score")[
        ["loss_func", "transform_name", "freeze_pretrained", "f_score", "lr"]
    ]
)
df_best_lr.to_csv("df_best_lr.csv")

print("Best with no augs")
print(df[df["transform_name"] == "no_augs"].tail(1).reset_index().to_dict())

print("Best config:", best_config)
print(best_config_spec)

shutil.copy(Path(best_config) / "config.toml", "config/best-predict-config.toml")
print(f"Config {best_config} was copied into config/best-predict-config.toml")

# %%
with open(best_config / "history.json", "r") as f:
    history = json.load(f)
fig_log, ax_log = plt.subplots()
ax_log.plot(history["train loss"], label="Training")
ax_log.plot(history["val loss"], label="Validation")
ax_log.legend()
ax_log.set_xlabel("Epoch")
ax_log.set_ylabel("Loss")
ax_log.set_yscale("log")
plt.tight_layout()
fig_log.savefig("loss_log.png", dpi=200, bbox_inches="tight")
fig_log.savefig("loss_log.pdf", bbox_inches="tight")
# plt.show()

# %%
fig_log, ax_log = plt.subplots()
ax_log.plot(1 - np.array(history["train f1"]), label="Training")
ax_log.plot(1 - np.array(history["val f1"]), label="Validation")
ax_log.legend()
ax_log.set_xlabel("Epoch")
ax_log.set_ylabel("1-F1")
ax_log.set_yscale("log")
plt.tight_layout()
fig_log.savefig("f1_log.png", dpi=200, bbox_inches="tight")
fig_log.savefig("f1_log.pdf", bbox_inches="tight")
# plt.show()

# plt.show()

# %%
print(
    df[
        [
            "transform_name",
            "signal_fraction",
            "loss_func",
            "lr",
            "focal_gamma",
            "f_score",
            "precision",
            "recall",
        ]
    ]
    .fillna(0)
    .sort_values("f_score")
    .to_string()
)

# %%
df.plot.scatter("precision", "recall")
plt.show()
