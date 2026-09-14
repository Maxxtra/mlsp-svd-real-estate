"""Compara un SVD custom cu numpy.linalg.svd pe matricea reala."""

import argparse
import os
import time
import csv
import importlib.util
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "results",
    "svd_validation.csv"
)


def load_func(path, name):
    spec = importlib.util.spec_from_file_location("custom", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return getattr(m, name)


def reconstruction_error(A, U, s, Vt):
    return np.linalg.norm(A - (U * s) @ Vt) / np.linalg.norm(A)


def orthogonality(M):
    return np.linalg.norm(M.T @ M - np.eye(M.shape[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--custom", required=True)
    ap.add_argument("--func", default="SVD")
    ap.add_argument("--A", default=os.path.join(DATA, "A.npy"))
    a = ap.parse_args()

    A = np.load(a.A)
    f = load_func(a.custom, a.func)

    # Custom
    t0 = time.time()
    U, S, V = f(A)
    t_custom = time.time() - t0

    s = np.diag(S) if np.ndim(S) == 2 else np.asarray(S)
    Vt = V.T

    # NumPy reference
    t0 = time.time()
    Un, sn, Vtn = np.linalg.svd(A, full_matrices=False)
    t_np = time.time() - t0

    # Numerical rank determined from reference SVD
    rank = np.linalg.matrix_rank(A)

    # Reconstruction
    rec = reconstruction_error(A, U, s, Vt)
    recn = reconstruction_error(A, Un, sn, Vtn)

    # Full U metric is kept for transparency
    orth_u_full = orthogonality(U)

    # Meaningful U metric: only non-null singular subspace
    orth_u_rank = orthogonality(U[:, :rank])
    orth_v_rank = orthogonality(V[:, :rank])

    orth_un_rank = orthogonality(Un[:, :rank])
    orth_vn_rank = orthogonality(Vtn[:rank, :].T)

    # Singular-value differences
    k = min(len(s), len(sn))
    sigma_diff_full = np.max(
        np.abs(np.sort(s)[::-1][:k] - sn[:k])
    )

    sigma_diff_rank = np.max(
        np.abs(np.sort(s)[::-1][:rank] - sn[:rank])
    )

    rows = [
        [
            "custom",
            A.shape[0],
            A.shape[1],
            rank,
            f"{rec:.3e}",
            f"{orth_u_full:.3e}",
            f"{orth_u_rank:.3e}",
            f"{orth_v_rank:.3e}",
            f"{sigma_diff_full:.3e}",
            f"{sigma_diff_rank:.3e}",
            f"{t_custom:.2f}",
        ],
        [
            "numpy",
            A.shape[0],
            A.shape[1],
            rank,
            f"{recn:.3e}",
            f"{orthogonality(Un):.3e}",
            f"{orth_un_rank:.3e}",
            f"{orth_vn_rank:.3e}",
            "0",
            "0",
            f"{t_np:.2f}",
        ],
    ]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "impl",
            "m",
            "n",
            "numerical_rank",
            "rel_reconstruction",
            "orth_U_full",
            "orth_U_rank",
            "orth_V_rank",
            "max_sigma_diff_full",
            "max_sigma_diff_rank",
            "seconds",
        ])
        w.writerows(rows)

    for r in rows:
        print(r)

    print("scris:", OUT)


if __name__ == "__main__":
    main()