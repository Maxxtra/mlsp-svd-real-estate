"""Acordul dintre metode: Jaccard pe perechi, cate proprietati sunt marcate de toate / de una singura.

    python src/agreement.py [--tag ""]
Citeste results/anomalies_*.csv, scrie results/agreement.csv
"""
import argparse, os, csv, glob, itertools, numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

def flagged(path):
    with open(path) as f: r = csv.DictReader(f); return {int(x["row"]) for x in r if x["flagged"] == "1"}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default=""); a = ap.parse_args()
    sets = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "results", f"anomalies_*{a.tag}.csv"))):
        name = os.path.basename(p)[len("anomalies_"):-len(".csv")]
        if a.tag and name.endswith(a.tag): name = name[:-len(a.tag)]
        sets[name] = flagged(p)
    if len(sets) < 2: print("am nevoie de cel putin 2 fisiere anomalies_*.csv"); return
    rows = []
    for x, y in itertools.combinations(sets, 2):
        j = len(sets[x] & sets[y]) / max(1, len(sets[x] | sets[y]))
        rows.append([f"{x}&{y}", len(sets[x]), len(sets[y]), len(sets[x] & sets[y]), f"{j:.4f}"])
        print(f"{x:<8} vs {y:<8} |x|={len(sets[x])} |y|={len(sets[y])} comune={len(sets[x]&sets[y])} Jaccard={j:.3f}")
    allc = set.intersection(*sets.values()); onlys = {k: len(v - set.union(*(s for kk, s in sets.items() if kk != k))) for k, v in sets.items()}
    print(f"marcate de toate {len(sets)} metodele: {len(allc)}"); print("doar de una:", onlys)
    with open(os.path.join(ROOT, "results", "agreement.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pair","n_a","n_b","common","jaccard"]); w.writerows(rows)
        w.writerow(["all_methods", "", "", len(allc), ""])
        for k, v in onlys.items(): w.writerow([f"only_{k}", "", "", v, ""])

if __name__ == "__main__":
    main()
