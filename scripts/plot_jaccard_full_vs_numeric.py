import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIGURES = os.path.join(ROOT, "figures")

pairs = ["IF vs KMeans", "IF vs SVD", "KMeans vs SVD"]

jaccard_full = [0.053, 0.050, 0.962]
jaccard_numeric = [0.235, 0.394, 0.138]

x = range(len(pairs))
width = 0.35

plt.figure(figsize=(9, 5))
plt.bar([i - width / 2 for i in x], jaccard_full, width=width, label="Full matrix")
plt.bar([i + width / 2 for i in x], jaccard_numeric, width=width, label="Numeric-only")

plt.xticks(list(x), pairs)
plt.ylabel("Jaccard similarity")
plt.ylim(0, 1.05)
plt.title("Agreement between anomaly detectors: full vs numeric-only features")
plt.legend()
plt.tight_layout()

out = os.path.join(FIGURES, "jaccard_full_vs_numeric.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()

print("scris:", out)