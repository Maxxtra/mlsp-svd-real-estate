import argparse, os, json, numpy as np, pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

NUM = ["TOTAL UNITS", "RESIDENTIAL UNITS", "COMMERCIAL UNITS", "GROSS SQUARE FEET", "LAND SQUARE FEET", "YEAR BUILT"]
CAT = ["NEIGHBORHOOD", "BUILDING CLASS CATEGORY", "BUILDING CLASS AT PRESENT"]

def load_clean(path, min_price=10_000, return_dropped=False):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    for c in NUM + ["SALE PRICE"]:
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",", "").str.replace("-", ""),
            errors="coerce"
        )

    df = df.dropna(subset=NUM + ["SALE PRICE"])

    # Intai aplicam filtrele structurale, independent de pret.
    df = df[
        (df["SALE PRICE"] > 0)
        & (df["TOTAL UNITS"] > 0)
        & (df["GROSS SQUARE FEET"] > 0)
        & (df["LAND SQUARE FEET"] > 0)
        & (df["YEAR BUILT"] > 1800)
    ].reset_index(drop=True)

    # Acum numaram exact cate tranzactii valide eliminam doar din cauza pragului de pret.
    dropped = int((df["SALE PRICE"] < min_price).sum())

    df = df[
        df["SALE PRICE"] >= min_price
    ].reset_index(drop=True)

    if return_dropped:
        return df, dropped

    return df

def standardize(X, robust=False):
    if robust:
        med = X.median()
        iqr = (X.quantile(0.75) - X.quantile(0.25)).replace(0, 1.0)
        return ((X - med) / iqr).to_numpy()

    mu = X.mean()
    sd = X.std(ddof=0).replace(0, 1.0)
    return ((X - mu) / sd).to_numpy()

def build_full(df, robust=False):
    X = pd.get_dummies(df[NUM + CAT], columns=CAT, drop_first=False).astype(float)
    A = standardize(X, robust)
    return A, list(X.columns)

def build_numeric(df, robust=False):
    X = df[NUM].astype(float)
    A = standardize(X, robust)
    return A, list(X.columns)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--robust", action="store_true")
    ap.add_argument("--drop-top-pct", type=float, default=0.0)
    ap.add_argument("--min-price", type=float, default=10_000)
    a = ap.parse_args()

    df, dropped = load_clean(
        a.csv,
        min_price=a.min_price,
        return_dropped=True
    )

    if a.drop_top_pct > 0:
        cut = df["SALE PRICE"].quantile(1 - a.drop_top_pct / 100)
        df = df[df["SALE PRICE"] <= cut].reset_index(drop=True)

    A_full, full_cols = build_full(df, a.robust)
    A_numeric, numeric_cols = build_numeric(df, a.robust)

    os.makedirs(DATA, exist_ok=True)

    np.save(os.path.join(DATA, "A.npy"), A_full)
    np.save(os.path.join(DATA, "A_numeric.npy"), A_numeric)
    np.save(os.path.join(DATA, "y.npy"), df["SALE PRICE"].to_numpy(float))

    meta = {
        "full_columns": full_cols,
        "numeric_columns": numeric_cols,
        "robust": a.robust,
        "drop_top_pct": a.drop_top_pct,
        "min_price": a.min_price,
        "rows_dropped_below_min_price": dropped,
        "m": A_full.shape[0],
        "n_full": A_full.shape[1],
        "n_numeric": A_numeric.shape[1]
    }

    with open(os.path.join(DATA, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("min sale price:", a.min_price)
    print("rows dropped below min price:", dropped)
    print("full:", A_full.shape)
    print("numeric:", A_numeric.shape)

if __name__ == "__main__":
    main()