import os

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".."
)

RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")


def plot_metrics():
    df = pd.read_csv(
        os.path.join(
            RESULTS,
            "regression_metrics.csv"
        )
    )

    labels = df["model"].tolist()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5)
    )

    axes[0].bar(
        labels,
        df["rmse"]
    )
    axes[0].set_title("RMSE by model")
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(
        axis="x",
        rotation=45
    )

    axes[1].bar(
        labels,
        df["mae"]
    )
    axes[1].set_title("MAE by model")
    axes[1].set_ylabel("MAE")
    axes[1].tick_params(
        axis="x",
        rotation=45
    )

    axes[2].bar(
        labels,
        df["r2"]
    )
    axes[2].set_title("R² by model")
    axes[2].set_ylabel("R²")
    axes[2].tick_params(
        axis="x",
        rotation=45
    )

    fig.tight_layout()

    out = os.path.join(
        FIGURES,
        "regression_metrics.png"
    )

    plt.savefig(
        out,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("scris:", out)


def plot_residual_distribution():
    df = pd.read_csv(
        os.path.join(
            RESULTS,
            "least_squares_top50_undervalued.csv"
        )
    )

    df_over = pd.read_csv(
        os.path.join(
            RESULTS,
            "least_squares_top50_overvalued.csv"
        )
    )

    all_residuals = pd.concat(
        [
            df[["residual"]],
            df_over[["residual"]]
        ],
        ignore_index=True
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        all_residuals["residual"],
        bins=25
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title(
        "Least squares residuals for top under/overvalued properties"
    )

    plt.tight_layout()

    out = os.path.join(
        FIGURES,
        "regression_residual_distribution.png"
    )

    plt.savefig(
        out,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("scris:", out)


def plot_neighborhood_concentration():
    df = pd.read_csv(
        os.path.join(
            RESULTS,
            "regression_neighborhood_concentration.csv"
        )
    )

    df = df[
        df["model"] == "least_squares"
    ]

    under = (
        df[
            df["group"] == "undervalued"
        ]
        .sort_values(
            "count",
            ascending=False
        )
        .head(10)
    )

    over = (
        df[
            df["group"] == "overvalued"
        ]
        .sort_values(
            "count",
            ascending=False
        )
        .head(10)
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    axes[0].barh(
        under["neighborhood"],
        under["count"]
    )

    axes[0].invert_yaxis()
    axes[0].set_title(
        "Top neighborhoods - undervalued"
    )
    axes[0].set_xlabel(
        "Count in top 50"
    )

    axes[1].barh(
        over["neighborhood"],
        over["count"]
    )

    axes[1].invert_yaxis()
    axes[1].set_title(
        "Top neighborhoods - overvalued"
    )
    axes[1].set_xlabel(
        "Count in top 50"
    )

    fig.tight_layout()

    out = os.path.join(
        FIGURES,
        "regression_neighborhood_concentration.png"
    )

    plt.savefig(
        out,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("scris:", out)


def main():
    os.makedirs(
        FIGURES,
        exist_ok=True
    )

    plot_metrics()
    plot_residual_distribution()
    plot_neighborhood_concentration()


if __name__ == "__main__":
    main()