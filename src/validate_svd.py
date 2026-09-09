"""Compara un SVD custom cu numpy.linalg.svd pe matricea reala.

    python src/validate_svd.py --custom src/custom_svd.py --func SVD
Functia custom trebuie sa intoarca (U, S, V) cu S matrice diagonala sau vector, si A ~ U S V^T.
"""
import argparse, os, time, csv, importlib.util, numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "svd_validation.csv")

def load_func(path, name):
    spec = importlib.util.spec_from_file_location("custom", path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return getattr(m, name)

def metrics(A, U, s, Vt):
    rec = np.linalg.norm(A - (U * s) @ Vt) / np.linalg.norm(A)
    ou = np.linalg.norm(U.T @ U - np.eye(U.shape[1])); ov = np.linalg.norm(Vt @ Vt.T - np.eye(Vt.shape[0]))
    return rec, ou, ov

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--custom", required=True); ap.add_argument("--func", default="SVD")
    ap.add_argument("--A", default=os.path.join(DATA, "A.npy"))
    a = ap.parse_args()
    A = np.load(a.A); f = load_func(a.custom, a.func)

    t0 = time.time(); U, S, V = f(A); t_custom = time.time() - t0
    s = np.diag(S) if np.ndim(S) == 2 else np.asarray(S); Vt = V.T
    t0 = time.time(); Un, sn, Vtn = np.linalg.svd(A, full_matrices=False); t_np = time.time() - t0

    rec, ou, ov = metrics(A, U, s, Vt); recn, oun, ovn = metrics(A, Un, sn, Vtn)
    k = min(len(s), len(sn)); ds = np.max(np.abs(np.sort(s)[::-1][:k] - sn[:k]))
    rows = [["custom", A.shape[0], A.shape[1], f"{rec:.3e}", f"{ou:.3e}", f"{ov:.3e}", f"{ds:.3e}", f"{t_custom:.2f}"],
            ["numpy",  A.shape[0], A.shape[1], f"{recn:.3e}", f"{oun:.3e}", f"{ovn:.3e}", "0", f"{t_np:.2f}"]]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["impl","m","n","rel_reconstruction","orth_U","orth_V","max_sigma_diff","seconds"]); w.writerows(rows)
    for r in rows: print(r)
    print("scris:", OUT)

if __name__ == "__main__":
    main()
