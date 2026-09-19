import os
import numpy as np

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest


ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

A = np.load(
    os.path.join(
        ROOT,
        "data",
        "A.npy"
    )
)

PCT = 95


def flags_from_scores(scores):
    threshold = np.percentile(
        scores,
        PCT
    )

    return scores > threshold


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


def kmeans_flags(seed):
    km = KMeans(
        n_clusters=4,
        random_state=seed,
        n_init=10
    ).fit(A)

    scores = np.linalg.norm(
        A - km.cluster_centers_[km.labels_],
        axis=1
    )

    return flags_from_scores(scores)


def iforest_flags(seed, n_estimators):
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=0.05,
        random_state=seed,
        n_jobs=-1
    ).fit(A)

    scores = -iso.score_samples(A)

    return flags_from_scores(scores)


def main():
    seeds = [
        42,
        7,
        13,
        123
    ]

    n_estimators_values = [
        100,
        500,
        1000,
        3000
    ]

    print("=== K-MEANS STABILITY ===")

    base = kmeans_flags(
        seeds[0]
    )

    for seed in seeds[1:]:
        current = kmeans_flags(
            seed
        )

        print(
            f"42 vs {seed}: "
            f"Jaccard={jaccard(base, current):.6f}"
        )

    print(
        "\n=== ISOLATION FOREST STABILITY ==="
    )

    for n_estimators in n_estimators_values:

        print(
            f"\n--- n_estimators = {n_estimators} ---"
        )

        base = iforest_flags(
            seed=seeds[0],
            n_estimators=n_estimators
        )

        values = []

        for seed in seeds[1:]:

            current = iforest_flags(
                seed=seed,
                n_estimators=n_estimators
            )

            jac = jaccard(
                base,
                current
            )

            values.append(jac)

            print(
                f"42 vs {seed}: "
                f"Jaccard={jac:.6f}"
            )

        print(
            f"Mean Jaccard: "
            f"{np.mean(values):.6f}"
        )

        print(
            f"Min Jaccard: "
            f"{np.min(values):.6f}"
        )

        print(
            f"Max Jaccard: "
            f"{np.max(values):.6f}"
        )


if __name__ == "__main__":
    main()