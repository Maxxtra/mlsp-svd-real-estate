"""Eroarea de reconstructie in functie de rangul k."""

import argparse
import os
import csv
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--kmax",
        type=int,
        default=0,
        help="0 = toate componentele"
    )
    ap.add_argument(
        "--A",
        default=os.path.join(ROOT, "data", "A.npy")
    )
    a = ap.parse_args()

    A = np.load(a.A)
    s = np.linalg.svd(A, compute_uv=False)

    energy = s ** 2
    info = np.cumsum(energy) / np.sum(energy)

    # Evita valori de forma -2e-16 din floating point
    remaining = np.clip(1.0 - info, 0.0, 1.0)
    rel_err = np.sqrt(remaining)

    k90 = int(np.searchsorted(info, 0.90) + 1)
    k95 = int(np.searchsorted(info, 0.95) + 1)

    numerical_rank = np.linalg.matrix_rank(A)

    if a.kmax <= 0:
        kmax = len(s)
    else:
        kmax = min(a.kmax, len(s))

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)

    with open(
        os.path.join(ROOT, "results", "rank_k.csv"),
        "w",
        newline=""
    ) as f:
        w = csv.writer(f)
        w.writerow([
            "k",
            "info_kept",
            "rel_reconstruction_error"
        ])

        for k in range(1, kmax + 1):
            w.writerow([
                k,
                f"{info[k - 1]:.6f}",
                f"{rel_err[k - 1]:.6f}"
            ])

    ks = np.arange(1, kmax + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(ks, rel_err[:kmax])

    if k90 <= kmax:
        plt.axvline(k90, linestyle="--")
        plt.text(
            k90,
            0.55,
            f"90% info: k={k90}",
            rotation=90,
            verticalalignment="center"
        )

    if k95 <= kmax:
        plt.axvline(k95, linestyle="--")
        plt.text(
            k95,
            0.40,
            f"95% info: k={k95}",
            rotation=90,
            verticalalignment="center"
        )

    plt.xlabel("Rank k")
    plt.ylabel("Relative reconstruction error")
    plt.title("Low-rank SVD approximation")
    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        os.path.join(ROOT, "figures", "rank_k.png"),
        dpi=200
    )

    print("numerical rank:", numerical_rank)
    print("k pentru 90%:", k90)
    print("k pentru 95%:", k95)
    print("scris results/rank_k.csv, figures/rank_k.png")


if __name__ == "__main__":
    main()