import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("optimise_postprocessing.csv")
df["f_score"] =  df["tp"] / (df["tp"]+0.5*(df["fp"]+df["fn"]))
df["m"] = (df["fp"] + df["fn"] +df["tp"]+df["tn"])
df["accuracy"] =  (df["tp"]+df["tn"]) / df["m"]
df["precision"] = df.tp / (df.tp+df.fp)
df["recall"] = df.tp / (df.tp+df.fn)
df["fpr"] = df.fp / (df.fp+df.tn)
for s in ("tp", "fp", "fn", "tn"):
    var = f"{s} / m"
    df[var] = df[s] / df.m
print(df.sort_values("f_score").tail()[[ "accuracy", "precision", "recall", "f_score", "fp / m", "miou", "kernel_size_grow", "kernel_size_denoise"]].round(3).to_latex())

#df.plot.scatter("kernel_size_grow", "f_score")
df.plot.scatter("kernel_size_denoise", "f_score")
df.plot.scatter("kernel_size_denoise", "fpr")
df.plot.scatter("kernel_size_denoise", "recall")
df.plot.scatter("kernel_size_denoise", "precision")
df.plot.scatter("kernel_size_grow", "recall")
df.plot.scatter("kernel_size_grow", "precision")
plt.show()





df = pd.read_csv("confusion_matrix.csv")
df["f_score"] =  df["tp"] / (df["tp"]+0.5*(df["fp"]+df["fn"]))
df["m"] = (df["fp"] + df["fn"] +df["tp"]+df["tn"])
df["accuracy"] =  (df["tp"]+df["tn"]) / df["m"]
df["precision"] = df.tp / (df.tp+df.fp)
df["recall"] = df.tp / (df.tp+df.fn)
df["fpr"] = df.fp / (df.fp+df.tn)


for s in ("tp", "fp", "fn", "tn"):
    var = f"{s} / m"
    df[var] = df[s] / df.m

print(df[["ds", "m", "tp / m", "tn / m", "fp / m", "fn / m"]].round(3).to_latex())
print(df[["ds", "accuracy", "precision", "recall", "f_score", "miou"]].round(3).to_latex())
