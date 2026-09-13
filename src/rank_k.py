import argparse
import csv
import os

import numpy as np


ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".."
)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--A",
        required=True
    )

    ap.add_argument(
        "--out",
        required=True
    )

    a = ap.parse_args()

    A = np.load(a.A)

    s = np.linalg.svd(
        A,
        compute_uv=False
    )

    energy = s ** 2

    info = np.cumsum(energy) / np.sum(energy)

    remaining = np.clip(
        1.0 - info,
        0.0,
        1.0
    )

    rel_err = np.sqrt(remaining)

    k90 = int(
        np.searchsorted(
            info,
            0.90
        ) + 1
    )

    k95 = int(
        np.searchsorted(
            info,
            0.95
        ) + 1
    )

    numerical_rank = np.linalg.matrix_rank(A)

    os.makedirs(
        os.path.dirname(a.out),
        exist_ok=True
    )

    with open(
        a.out,
        "w",
        newline=""
    ) as f:
        w = csv.writer(f)

        w.writerow([
            "k",
            "info_kept",
            "rel_reconstruction_error"
        ])

        for k in range(
            1,
            len(s) + 1
        ):
            w.writerow([
                k,
                f"{info[k - 1]:.6f}",
                f"{rel_err[k - 1]:.6f}"
            ])

    print("shape:", A.shape)
    print("numerical rank:", numerical_rank)
    print("k pentru 90%:", k90)
    print("k pentru 95%:", k95)
    print("scris:", a.out)


if __name__ == "__main__":
    main()