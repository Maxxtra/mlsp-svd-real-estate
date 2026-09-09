"""Eroarea de reconstructie in functie de rangul k, si k-ul pentru 90% / 95% informatie.

    python src/rank_k.py [--kmax 50]
Scrie results/rank_k.csv si figures/rank_k.png
"""
import argparse, os, csv, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kmax", type=int, default=50)
    ap.add_argument("--A", default=os.path.join(ROOT, "data", "A.npy")); a = ap.parse_args()
    A = np.load(a.A); s = np.linalg.svd(A, compute_uv=False)
    info = np.cumsum(s**2) / np.sum(s**2); rel_err = np.sqrt(1 - info)
    k90 = int(np.argmax(info >= .90) + 1); k95 = int(np.argmax(info >= .95) + 1)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True); os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "rank_k.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["k","info_kept","rel_reconstruction_error"])
        for k in range(1, min(a.kmax, len(s)) + 1): w.writerow([k, f"{info[k-1]:.4f}", f"{rel_err[k-1]:.4f}"])
    ks = np.arange(1, min(a.kmax, len(s)) + 1)
    plt.figure(figsize=(6,4)); plt.plot(ks, rel_err[:len(ks)], marker="o", ms=3)
    plt.axvline(k90, ls="--", c="gray"); plt.axvline(k95, ls="--", c="gray")
    plt.text(k90, rel_err[0]*.9, f"90% @ k={k90}"); plt.text(k95, rel_err[0]*.8, f"95% @ k={k95}")
    plt.xlabel("rang k"); plt.ylabel("eroare relativa de reconstructie"); plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "figures", "rank_k.png"), dpi=150)
    print(f"k pentru 90%: {k90}, pentru 95%: {k95}. scris results/rank_k.csv, figures/rank_k.png")

if __name__ == "__main__":
    main()
