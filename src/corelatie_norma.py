import sys, numpy as np, csv

matrice = sys.argv[1] if len(sys.argv) > 1 else "data/A.npy"
tag = sys.argv[2] if len(sys.argv) > 2 else ""

A = np.load(matrice)
norma = np.linalg.norm(A, axis=1)                          # cat de "mare" e fiecare rand

print(f"{matrice}")
for m in ["svd", "kmeans", "iforest"]:
    scor = np.array([float(r["score"]) for r in csv.DictReader(open(f"results/anomalies_{m}{tag}.csv"))])
    print(f"{m:<8} corelatie scor vs norma randului: {np.corrcoef(scor, norma)[0, 1]:.3f}")
    if m == "svd":
        print(f"{'svd (radical)':<8} corelatie scor vs norma randului: {np.corrcoef(np.sqrt(scor), norma)[0, 1]:.3f}")
