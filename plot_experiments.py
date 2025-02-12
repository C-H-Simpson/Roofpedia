from pathlib import Path
import pandas as pd
import toml
import json
import numpy as np
import matplotlib.pyplot as plt

paths = Path("/home/ucbqc38/Scratch/experiments").glob("experiment_*")
results = []
best_f1 = 0
best_config = ""
for p in paths:
    config = toml.load(p/"config.toml")
    with open(p/"history.json", "r") as f:
        history=json.load(f)
    #print(config)
    #if config["model_path"]:
        #continue
    #if config["freeze_pretrained"] == 0:
        #continue
    label = f"{config['model_path']} ; {config['freeze_pretrained']}; {config['transform']}; {config['lr']}"
    for key in history:
        history[key] = np.array(history[key])
    f_score = history["val tp"] / (history["val tp"]+0.5*(history["val fp"]+history["val fn"]))
    f_score_train = history["train tp"] / (history["train tp"]+0.5*(history["train fp"]+history["train fn"]))
    config["f_score"] = f_score[-1]
    config["miou"] = history["val miou"][-1]
    results.append(config)
    y = 1 - f_score
    y = f_score_train - f_score
    #y = accuracy_n
    plt.plot(y, label=label)
    plt.text(len(y)-1, y[-1], label)
    plt.yscale("log")
    if f_score[-1] > best_f1:
        best_f1 = f_score[-1]
        best_config=p

#df = pd.DataFrame(results).sort_values("f_score")
#df.to_csv("results.csv")
#print(df.tail())

print("Best config:", best_config)
plt.show()

