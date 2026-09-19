import os, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")

raw = pd.read_csv(os.path.join(RESULTS, "regression_neighborhood_concentration.csv"))
log = pd.read_csv(os.path.join(RESULTS, "regression_neighborhood_concentration_log.csv"))

raw = raw[raw["model"] == "ridge_alpha_1000"]
log = log[log["model"] == "ridge_alpha_1000"]

def top_counts(df, group_name, top_n=10):
    df = df[df["group"] == group_name][["neighborhood", "count"]].copy()
    return dict(zip(df["neighborhood"], df["count"]))

def make_plot(group_name, title_suffix, out_name):
    raw_counts = top_counts(raw, group_name)
    log_counts = top_counts(log, group_name)

    neighborhoods = sorted(set(raw_counts) | set(log_counts))

    combined = []
    for n in neighborhoods:
        combined.append(
            (
                n,
                raw_counts.get(n, 0),
                log_counts.get(n, 0)
            )
        )

    combined.sort(key=lambda x: max(x[1], x[2]), reverse=True)
    combined = combined[:10]

    labels = [x[0] for x in combined]
    raw_vals = [x[1] for x in combined]
    log_vals = [x[2] for x in combined]

    x = range(len(labels))
    width = 0.4

    plt.figure(figsize=(12, 6))
    plt.bar([i - width / 2 for i in x], raw_vals, width=width, label="Raw price")
    plt.bar([i + width / 2 for i in x], log_vals, width=width, label="Log price")

    plt.xticks(list(x), labels, rotation=45, ha="right")
    plt.ylabel("Count in top 50")
    plt.title(f"Ridge alpha=1000: {title_suffix}")
    plt.legend()
    plt.tight_layout()

    out = os.path.join(FIGURES, out_name)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()

    print("scris:", out)

make_plot(
    "undervalued",
    "top neighborhoods - undervalued (raw vs log)",
    "ridge1000_neighborhoods_undervalued_raw_vs_log.png"
)

make_plot(
    "overvalued",
    "top neighborhoods - overvalued (raw vs log)",
    "ridge1000_neighborhoods_overvalued_raw_vs_log.png"
)