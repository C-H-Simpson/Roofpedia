"""
From already calculated confusion matrix results, make a nicely formatted table.
"""
# %%
import pandas as pd
import numpy as np

df = pd.read_csv("confusion_matrix.csv")
# Join the different training sets together.
df = pd.concat(
    (
        df,
        df[df.ds.str.contains("training")][["kfold", "tp", "fp", "fn", "tn"]]
        .groupby("kfold")
        .sum()
        .assign(ds="training")
        .reset_index(),
    )
)
df = df.set_index(["ds", "kfold"]).sort_index()

df["f_score"] = df["tp"] / (df["tp"] + 0.5 * (df["fp"] + df["fn"]))
df["m"] = df["fp"] + df["fn"] + df["tp"] + df["tn"]
df["accuracy"] = (df["tp"] + df["tn"]) / df["m"]
df["precision"] = df.tp / (df.tp + df.fp)
df["recall"] = df.tp / (df.tp + df.fn)
df["fpr"] = df.fp / (df.fp + df.tn)
df["miou"] = np.nanmean(
    [
        df.tn / (df.tn + df.fn + df.fp),
        df.tp / (df.tp + df.fn + df.fp),
    ],
    axis=0,
)


for s in ("tp", "fp", "fn", "tn"):
    var = f"{s} / m"
    df[var] = df[s] / df.m

print(
    df.loc[["training", "validation", "testing"]][
        ["m", "tp / m", "tn / m", "fp / m", "fn / m"]
    ]
    .round(3)
    .to_latex()
)
print(
    df.loc[["training", "validation", "testing"]][
        ["accuracy", "precision", "recall", "f_score", "miou"]
    ]
    .round(3)
    .to_latex()
)
