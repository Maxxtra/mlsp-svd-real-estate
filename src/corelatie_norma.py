"""Corelatia dintre scorul fiecarui detector si norma randului.

    python src/corelatie_norma.py                                 # matricea completa
    python src/corelatie_norma.py data/A_numeric.npy _numeric     # doar caracteristicile numerice
Scrie results/corelatie_norma<tag>.csv
"""
import sys, os, numpy as np, csv

matrice = sys.argv[1] if len(sys.argv) > 1 else "data/A.npy"
tag = sys.argv[2] if len(sys.argv) > 2 else ""

A = np.load(matrice)
norma = np.linalg.norm(A, axis=1)                          # cat de "mare" e fiecare rand

randuri = []
for m in ["svd", "kmeans", "iforest"]:
    scor = np.array([float(r["score"]) for r in csv.DictReader(open(f"results/anomalies_{m}{tag}.csv"))])
    randuri.append([m, f"{np.corrcoef(scor, norma)[0, 1]:.3f}"])
    if m == "svd":                                         # scorul svd e patratic, cel kmeans e o distanta
        randuri.append(["svd_radical", f"{np.corrcoef(np.sqrt(scor), norma)[0, 1]:.3f}"])

print(matrice)
for nume, c in randuri:
    print(f"{nume:<12} corelatie scor vs norma randului: {c}")

iesire = f"results/corelatie_norma{tag}.csv"
with open(iesire, "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["matrice", "metoda", "corelatie_cu_norma"])
    for nume, c in randuri:
        w.writerow([os.path.basename(matrice), nume, c])
print("scris:", iesire)
