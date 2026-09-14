import argparse
import os

import numpy as np
import pandas as pd

from build_matrix import load_clean


ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".."
)

DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")


def load_data(csv_path, A_path, y_path):
    A = np.load(A_path)
    y = np.load(y_path)

    df = load_clean(csv_path)

    if len(df) != len(A) or len(y) != len(A):
        raise ValueError(
            f"Dimensiuni incompatibile: df={len(df)}, A={len(A)}, y={len(y)}"
        )

    return A, y, df


def compute_svd(A, tol=1e-10):
    U, s, Vt = np.linalg.svd(
        A,
        full_matrices=False
    )

    if len(s) == 0:
        return U, s, Vt, 0

    cutoff = tol * s[0]

    rank = int(
        np.sum(s > cutoff)
    )

    return (
        U[:, :rank],
        s[:rank],
        Vt[:rank, :],
        rank
    )


def predict_least_squares(
    U,
    s,
    Vt,
    y_centered,
    y_mean
):
    coefficients = (
        Vt.T
        @ (
            (U.T @ y_centered)
            / s
        )
    )

    predictions = (
        U
        @ (
            U.T @ y_centered
        )
        + y_mean
    )

    return coefficients, predictions


def predict_ridge(
    U,
    s,
    Vt,
    y_centered,
    y_mean,
    alpha
):
    projected_y = (
        U.T @ y_centered
    )

    factors = (
        s
        / (
            s ** 2
            + alpha
        )
    )

    coefficients = (
        Vt.T
        @ (
            factors
            * projected_y
        )
    )

    shrinkage = (
        s ** 2
        / (
            s ** 2
            + alpha
        )
    )

    predictions = (
        U
        @ (
            shrinkage
            * projected_y
        )
        + y_mean
    )

    return coefficients, predictions


def compute_metrics(
    y,
    predictions
):
    residuals = (
        y - predictions
    )

    mse = np.mean(
        residuals ** 2
    )

    rmse = np.sqrt(mse)

    mae = np.mean(
        np.abs(residuals)
    )

    ss_res = np.sum(
        residuals ** 2
    )

    ss_tot = np.sum(
        (
            y - np.mean(y)
        ) ** 2
    )

    r2 = (
        1.0
        - ss_res / ss_tot
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


def make_residual_table(
    df,
    y,
    predictions,
    model_name
):
    residuals = (
        y - predictions
    )

    result = pd.DataFrame({
        "row_id": np.arange(len(df)),
        "model": model_name,
        "sale_price": y,
        "predicted_price": predictions,
        "residual": residuals,
        "absolute_residual": np.abs(residuals)
    })

    metadata_columns = [
        "BOROUGH",
        "NEIGHBORHOOD",
        "BUILDING CLASS CATEGORY",
        "ADDRESS",
        "ZIP CODE"
    ]

    for column in metadata_columns:
        if column in df.columns:
            result[column] = (
                df[column]
                .reset_index(drop=True)
            )

    return result


def save_top_residuals(
    residual_table,
    model_name
):
    undervalued = (
        residual_table
        .sort_values(
            "residual",
            ascending=True
        )
        .head(50)
    )

    overvalued = (
        residual_table
        .sort_values(
            "residual",
            ascending=False
        )
        .head(50)
    )

    undervalued.to_csv(
        os.path.join(
            RESULTS,
            f"{model_name}_top50_undervalued.csv"
        ),
        index=False
    )

    overvalued.to_csv(
        os.path.join(
            RESULTS,
            f"{model_name}_top50_overvalued.csv"
        ),
        index=False
    )

    return undervalued, overvalued


def neighborhood_summary(
    undervalued,
    overvalued,
    model_name
):
    rows = []

    for group_name, group in [
        ("undervalued", undervalued),
        ("overvalued", overvalued)
    ]:
        counts = (
            group["NEIGHBORHOOD"]
            .value_counts()
        )

        for neighborhood, count in counts.items():
            rows.append({
                "model": model_name,
                "group": group_name,
                "neighborhood": neighborhood,
                "count": int(count),
                "share": count / 50.0
            })

    return rows


def run_model(
    model_name,
    predictions,
    y,
    df,
    rank,
    alpha=None
):
    model_metrics = compute_metrics(
        y,
        predictions
    )

    residual_table = (
        make_residual_table(
            df,
            y,
            predictions,
            model_name
        )
    )

    undervalued, overvalued = (
        save_top_residuals(
            residual_table,
            model_name
        )
    )

    neighborhood_rows = (
        neighborhood_summary(
            undervalued,
            overvalued,
            model_name
        )
    )

    metric_row = {
        "model": model_name,
        "alpha": (
            ""
            if alpha is None
            else alpha
        ),
        "rank": rank,
        "rmse": model_metrics["rmse"],
        "mae": model_metrics["mae"],
        "r2": model_metrics["r2"]
    }

    return (
        metric_row,
        neighborhood_rows
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "csv"
    )

    parser.add_argument(
        "--A",
        default=os.path.join(
            DATA,
            "A.npy"
        )
    )

    parser.add_argument(
        "--y",
        default=os.path.join(
            DATA,
            "y.npy"
        )
    )

    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[
            100.0,
            1000.0,
            10000.0,
            100000.0
        ]
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=1e-10
    )

    parser.add_argument(
        "--min-sale-price",
        type=float,
        default=10000.0
    )

    args = parser.parse_args()

    os.makedirs(
        RESULTS,
        exist_ok=True
    )

    A, y, df = load_data(
        args.csv,
        args.A,
        args.y
    )

    mask = (
        y >= args.min_sale_price
    )

    A = A[mask]
    y = y[mask]
    df = (
        df.loc[mask]
        .reset_index(drop=True)
    )

    print("A:", A.shape)
    print("y:", y.shape)
    print(
        "min sale price:",
        args.min_sale_price
    )

    y_mean = np.mean(y)

    y_centered = (
        y - y_mean
    )

    U, s, Vt, rank = compute_svd(
        A,
        args.tol
    )

    print(
        "numerical rank:",
        rank
    )

    metric_rows = []
    neighborhood_rows = []

    _, ls_predictions = (
        predict_least_squares(
            U,
            s,
            Vt,
            y_centered,
            y_mean
        )
    )

    (
        ls_metric_row,
        ls_neighborhood_rows
    ) = run_model(
        "least_squares",
        ls_predictions,
        y,
        df,
        rank
    )

    metric_rows.append(
        ls_metric_row
    )

    neighborhood_rows.extend(
        ls_neighborhood_rows
    )

    print(
        "least_squares:",
        f"RMSE={ls_metric_row['rmse']:.3f}",
        f"MAE={ls_metric_row['mae']:.3f}",
        f"R2={ls_metric_row['r2']:.6f}"
    )

    for alpha in args.alphas:
        model_name = (
            f"ridge_alpha_{alpha:g}"
        )

        _, predictions = (
            predict_ridge(
                U,
                s,
                Vt,
                y_centered,
                y_mean,
                alpha
            )
        )

        (
            metric_row,
            model_neighborhood_rows
        ) = run_model(
            model_name,
            predictions,
            y,
            df,
            rank,
            alpha
        )

        metric_rows.append(
            metric_row
        )

        neighborhood_rows.extend(
            model_neighborhood_rows
        )

        print(
            model_name + ":",
            f"RMSE={metric_row['rmse']:.3f}",
            f"MAE={metric_row['mae']:.3f}",
            f"R2={metric_row['r2']:.6f}"
        )

    pd.DataFrame(
        metric_rows
    ).to_csv(
        os.path.join(
            RESULTS,
            "regression_metrics.csv"
        ),
        index=False
    )

    pd.DataFrame(
        neighborhood_rows
    ).to_csv(
        os.path.join(
            RESULTS,
            "regression_neighborhood_concentration.csv"
        ),
        index=False
    )

    print(
        "scris results/regression_metrics.csv"
    )

    print(
        "scris results/regression_neighborhood_concentration.csv"
    )

    print(
        "scrise fisierele top 50 undervalued/overvalued pentru fiecare model"
    )


if __name__ == "__main__":
    main()