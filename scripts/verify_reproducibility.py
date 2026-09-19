import hashlib
import os

import numpy as np

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

import sys

ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.insert(
    0,
    os.path.join(ROOT, "src")
)

from custom_svd import SVD


A_PATH = os.path.join(
    ROOT,
    "data",
    "A.npy"
)

Y_PATH = os.path.join(
    ROOT,
    "data",
    "y.npy"
)

KS = [3, 5, 10, 50, 321]
PCT = 95


def hash_indices(indices):
    text = ",".join(
        str(int(x))
        for x in indices
    )

    return hashlib.sha256(
        text.encode()
    ).hexdigest()[:16]


def anomaly_scores_custom(
    A,
    U,
    S,
    V,
    k
):
    s = np.diag(S)

    Ak = (
        U[:, :k]
        * s[:k]
    ) @ V[:, :k].T

    return np.sum(
        (A - Ak) ** 2,
        axis=1
    )


def anomaly_scores_numpy(
    A,
    U,
    s,
    Vt,
    k
):
    Ak = (
        U[:, :k]
        * s[:k]
    ) @ Vt[:k, :]

    return np.sum(
        (A - Ak) ** 2,
        axis=1
    )


def flags_from_scores(
    scores,
    pct=95
):
    threshold = np.percentile(
        scores,
        pct
    )

    return (
        scores > threshold,
        threshold
    )


def jaccard(a, b):
    intersection = np.sum(
        a & b
    )

    union = np.sum(
        a | b
    )

    return (
        intersection / union
        if union
        else 1.0
    )


def compare_custom_numpy(
    A,
    Uc,
    Sc,
    Vc,
    Un,
    sn,
    Vtn
):
    print(
        "\n=== CUSTOM SVD vs NUMPY ==="
    )

    for k in KS:
        custom_scores = (
            anomaly_scores_custom(
                A,
                Uc,
                Sc,
                Vc,
                k
            )
        )

        numpy_scores = (
            anomaly_scores_numpy(
                A,
                Un,
                sn,
                Vtn,
                k
            )
        )

        custom_flags, custom_thr = (
            flags_from_scores(
                custom_scores,
                PCT
            )
        )

        numpy_flags, numpy_thr = (
            flags_from_scores(
                numpy_scores,
                PCT
            )
        )

        custom_indices = np.where(
            custom_flags
        )[0]

        numpy_indices = np.where(
            numpy_flags
        )[0]

        same_flags = np.array_equal(
            custom_flags,
            numpy_flags
        )

        different = int(
            np.sum(
                custom_flags
                != numpy_flags
            )
        )

        corr = np.corrcoef(
            custom_scores,
            numpy_scores
        )[0, 1]

        max_abs_diff = np.max(
            np.abs(
                custom_scores
                - numpy_scores
            )
        )

        print(
            f"\nk={k}"
        )

        print(
            "custom anomalies:",
            len(custom_indices)
        )

        print(
            "numpy anomalies: ",
            len(numpy_indices)
        )

        print(
            "identical lists:",
            same_flags
        )

        print(
            "different flags:",
            different
        )

        print(
            "Jaccard:",
            f"{jaccard(custom_flags, numpy_flags):.6f}"
        )

        print(
            "score correlation:",
            f"{corr:.12f}"
        )

        print(
            "max abs score diff:",
            f"{max_abs_diff:.12e}"
        )

        print(
            "custom hash:",
            hash_indices(
                custom_indices
            )
        )

        print(
            "numpy hash: ",
            hash_indices(
                numpy_indices
            )
        )

        print(
            "custom threshold:",
            custom_thr
        )

        print(
            "numpy threshold: ",
            numpy_thr
        )


def verify_three_detectors(
    A,
    y,
    Uc,
    Sc,
    Vc,
    k=5
):
    print(
        "\n=== THREE DETECTORS, k=5 ==="
    )

    svd_scores = (
        anomaly_scores_custom(
            A,
            Uc,
            Sc,
            Vc,
            k
        )
    )

    svd_flags, _ = (
        flags_from_scores(
            svd_scores,
            PCT
        )
    )

    km = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    ).fit(A)

    kmeans_scores = np.linalg.norm(
        A
        - km.cluster_centers_[
            km.labels_
        ],
        axis=1
    )

    kmeans_flags, _ = (
        flags_from_scores(
            kmeans_scores,
            PCT
        )
    )

    iso = IsolationForest(
        contamination=0.05,
        random_state=42
    ).fit(A)

    iforest_scores = (
        -iso.score_samples(A)
    )

    iforest_flags, _ = (
        flags_from_scores(
            iforest_scores,
            PCT
        )
    )

    print(
        "SVD anomalies:",
        int(
            np.sum(svd_flags)
        )
    )

    print(
        "K-Means anomalies:",
        int(
            np.sum(kmeans_flags)
        )
    )

    print(
        "Isolation Forest anomalies:",
        int(
            np.sum(iforest_flags)
        )
    )

    print(
        "\nK-Means vs Isolation Forest:",
        f"{jaccard(kmeans_flags, iforest_flags):.6f}"
    )

    print(
        "Isolation Forest vs SVD:",
        f"{jaccard(iforest_flags, svd_flags):.6f}"
    )

    print(
        "K-Means vs SVD:",
        f"{jaccard(kmeans_flags, svd_flags):.6f}"
    )

    common = (
        svd_flags
        & kmeans_flags
        & iforest_flags
    )

    common_indices = np.where(
        common
    )[0]

    print(
        "\ncommon all three:",
        len(common_indices)
    )

    print(
        "common hash:",
        hash_indices(
            common_indices
        )
    )

    print(
        "median price common:",
        np.median(
            y[common]
        )
    )

    print(
        "global median price:",
        np.median(y)
    )


def main():
    A = np.load(A_PATH)
    y = np.load(Y_PATH)

    print(
        "A shape:",
        A.shape
    )

    print(
        "y shape:",
        y.shape
    )

    print(
        "\nComputing custom SVD..."
    )

    Uc, Sc, Vc = SVD(A)

    print(
        "Computing NumPy SVD..."
    )

    Un, sn, Vtn = np.linalg.svd(
        A,
        full_matrices=False
    )

    compare_custom_numpy(
        A,
        Uc,
        Sc,
        Vc,
        Un,
        sn,
        Vtn
    )

    verify_three_detectors(
        A,
        y,
        Uc,
        Sc,
        Vc,
        k=5
    )


if __name__ == "__main__":
    main()