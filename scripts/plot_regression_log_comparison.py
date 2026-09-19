import os, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")

raw = pd.read_csv(os.path.join(RESULTS, "regression_metrics.csv"))
log = pd.read_csv(os.path.join(RESULTS, "regression_metrics_log.csv"))

raw = raw[raw["model"] == "ridge_alpha_1000"].iloc[0]
log = log[log["model"] == "ridge_alpha_1000"].iloc[0]

metrics = ["rmse", "mae", "r2"]
titles = ["RMSE", "MAE", "R²"]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for i, metric in enumerate(metrics):
    axes[i].bar(
        ["Raw price", "Log price"],
        [raw[metric], log[metric]]
    )
    axes[i].set_title(titles[i])

fig.suptitle("Ridge alpha=1000: raw vs log-transformed sale price")
fig.tight_layout()

out = os.path.join(
    FIGURES,
    "regression_raw_vs_log.png"
)

plt.savefig(
    out,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("scris:", out)