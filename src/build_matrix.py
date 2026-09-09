"""Construieste matricea de caracteristici exact ca in raportul Mariei.

    python src/build_matrix.py data/nyc-rolling-sales.csv [--robust] [--drop-top-pct 1]
Scrie data/A.npy (standardizat), data/y.npy (SALE PRICE), data/meta.json (numele coloanelor).
"""
import argparse, os, json, numpy as np, pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
NUM = ["TOTAL UNITS","RESIDENTIAL UNITS","COMMERCIAL UNITS","GROSS SQUARE FEET","LAND SQUARE FEET","YEAR BUILT"]
CAT = ["NEIGHBORHOOD","BUILDING CLASS CATEGORY","BUILDING CLASS AT PRESENT"]

def load_clean(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for c in NUM + ["SALE PRICE"]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.replace("-", ""), errors="coerce")
    df = df.dropna(subset=NUM + ["SALE PRICE"])
    df = df[(df["SALE PRICE"] > 0) & (df["TOTAL UNITS"] > 0) & (df["GROSS SQUARE FEET"] > 0)
            & (df["LAND SQUARE FEET"] > 0) & (df["YEAR BUILT"] > 1800)]
    return df.reset_index(drop=True)

def build(df, robust=False):
    X = pd.get_dummies(df[NUM + CAT], columns=CAT, drop_first=False).astype(float)
    if robust:
        med = X.median(); iqr = (X.quantile(.75) - X.quantile(.25)).replace(0, 1.0)
        A = ((X - med) / iqr).to_numpy()
    else:
        mu = X.mean(); sd = X.std(ddof=0).replace(0, 1.0)
        A = ((X - mu) / sd).to_numpy()
    return A, list(X.columns)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--robust", action="store_true")
    ap.add_argument("--drop-top-pct", type=float, default=0.0)
    a = ap.parse_args()
    df = load_clean(a.csv)
    if a.drop_top_pct > 0:
        cut = df["SALE PRICE"].quantile(1 - a.drop_top_pct/100); df = df[df["SALE PRICE"] <= cut].reset_index(drop=True)
    A, cols = build(df, a.robust)
    os.makedirs(DATA, exist_ok=True)
    np.save(os.path.join(DATA, "A.npy"), A); np.save(os.path.join(DATA, "y.npy"), df["SALE PRICE"].to_numpy(float))
    json.dump({"columns": cols, "robust": a.robust, "drop_top_pct": a.drop_top_pct, "m": A.shape[0], "n": A.shape[1]},
              open(os.path.join(DATA, "meta.json"), "w"), indent=2)
    print(f"A: {A.shape}, salvat in data/")

if __name__ == "__main__":
    main()
