# mlsp-svd-real-estate

Lucrarea 3 din grupul MLSP: un pipeline SVD implementat de la zero, aplicat pe ~29.000 de tranzacții
imobiliare din New York. Întrebarea de paper: poate detecta proprietăți greșit evaluate la fel de bine
ca metodele de bibliotecă, și cât de mult sunt de acord SVD, K-Means și Isolation Forest când marchează
aceleași proprietăți drept anomalii?

Echipa: Maria (nucleul SVD și modelarea prețului), David (detectoarele de anomalii, acordul dintre ele,
sensibilitate, robustețe, reproducibilitate).

Punctul de plecare e proiectul Mariei: https://github.com/mariatimbus/NYC_RealEstate (SVD-ul custom e
în `src/custom_svd.py` acolo). Datele: https://www.kaggle.com/datasets/new-york-city/nyc-property-sales

## Setup (5 minute)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# pune nyc-rolling-sales.csv de pe Kaggle in data/
python src/build_matrix.py data/nyc-rolling-sales.csv     # -> data/A.npy, data/y.npy, data/meta.json
```

Nu ai nevoie de GPU. Totul rulează pe laptop.

## Structura

```
src/build_matrix.py      curatare + one-hot + standardizare, exact ca in raportul Mariei -> A (m x n)
src/validate_svd.py      compara un SVD custom cu numpy: reconstructie, ortogonalitate, timp      (Maria)
src/rank_k.py            eroarea de reconstructie in functie de k, pragurile 90% / 95%            (Maria)
src/anomaly_detectors.py trei scoruri: SVD reconstruction, K-Means distance, Isolation Forest    (David)
src/agreement.py         Jaccard intre liste, cate proprietati sunt marcate de toate metodele    (David)
results/                 CSV-uri si liste de proprietati
figures/                 figurile, generate din scripturi
```

## Tutorial 1 (Maria, până vineri): validarea SVD-ului tău pe matricea reală

Copiază `custom_svd.py` din repo-ul tău în `src/`, apoi:

```bash
python src/validate_svd.py --custom src/custom_svd.py --func SVD
```

Scriptul încarcă `data/A.npy` (29275×408), rulează SVD-ul tău și `numpy.linalg.svd`, și scrie în
`results/svd_validation.csv`: eroarea relativă de reconstrucție ‖A − UΣVᵀ‖/‖A‖, ‖UᵀU − I‖, ‖VᵀV − I‖,
diferența maximă între valorile singulare, timpul fiecăruia. În raportul tău ai făcut asta pe o matrice
de test de 180×24; acum o facem pe cea reală. Cifrele intră în abstract.

Următorul pas (luni): `python src/rank_k.py`, curba erorii în funcție de k și k-ul la care păstrezi
90% și 95% din informație. Apoi regresia (joi), într-un fișier nou `src/regression.py`.

## Tutorial 2 (David, până vineri): două detectoare fără SVD

```bash
python src/anomaly_detectors.py --methods kmeans iforest --pct 95
```

Rulează K-Means (k=4, scor = distanța la centroidul propriu) și Isolation Forest pe `A`, marchează ca
anomalii proprietățile peste percentila 95 a fiecărui scor, și scrie `results/anomalies_<metoda>.csv`
(indexul rândului, scorul, marcat/nu). Asta e livrabilul tău de vineri și nu depinde de Maria.

De luni adaugi `--methods svd`, care folosește SVD-ul Mariei (eroarea de reconstrucție la rang k).
Apoi:

```bash
python src/agreement.py          # Jaccard pe fiecare pereche, cate sunt marcate de toate trei
python src/anomaly_detectors.py --methods svd kmeans iforest --pct 90    # sensibilitate la prag
python src/anomaly_detectors.py --methods svd --k 3                      # sensibilitate la rang
```

Robustețea (duminică): rulezi tot fără top 1% cele mai scumpe tranzacții (`--drop-top-pct 1`) și cu
standardizare robustă (`build_matrix.py --robust`), și vezi cât se schimbă listele.

De citit: Isolation Forest, Liu et al. 2008, https://ieeexplore.ieee.org/document/4781136

## Rezultatele

Un experiment e gata când scriptul îl reproduce de la zero și CSV-ul sau figura e comisă în `results/`
sau `figures/`. Nu „merge la mine în notebook".
