# David, planul tău până pe 27 septembrie

Attack Zoo + SVD imobiliare. Lista de mai jos e a ta: o iei de sus în jos, fiecare pas are termenul lui, care e întâlnirea la
care vreau să-l văd făcut. Primul pas e gândit să-l termini singur, fără să aștepți după nimeni.

*Attack Zoo + SVD imobiliare · evaluare și reproducibilitate pe amândouă · repo-uri: [github.com/Maxxtra/mlsp-attack-zoo](https://github.com/Maxxtra/mlsp-attack-zoo), [github.com/Maxxtra/mlsp-svd-real-estate](https://github.com/Maxxtra/mlsp-svd-real-estate)*

### Pe Attack Zoo

1. **până vineri 11** Task-ul tău e separat de al Andreei: tu iei atacurile „grele". Instalezi
   [AutoAttack](https://github.com/fra31/auto-attack) (referința standard
   din domeniu, citește [lucrarea](https://arxiv.org/abs/2003.01690)) și
   rulezi AutoAttack pe ResNet-50 pe CIFAR-10 la ε = 8/255. Scoți acuratețea curată și cea robustă.
   Asta e cifra ta de vineri și e independentă de ce face Andreea. Vă aliniați luni pe același
   script de încărcat date, ca să fie comparabile.
2. **până luni 14** Adaugi [Carlini-Wagner](https://arxiv.org/abs/1608.04644)
   din torchattacks și rulezi ambele (CW, AutoAttack) pe toate 4 modelele pe CIFAR-10. Coloanele
   tale intră în tabelul Andreei.
3. **până joi 17** Matricea de transfer 4×4: generezi exemple adversariale cu PGD pe modelul A
   și le evaluezi pe B, C, D. 16 celule, diagonala e atacul direct.
4. **până duminică 20** Curbele tărie-vs-iterații (PGD cu 1, 5, 10, 20, 50 de iterații pe
   fiecare model) și vizualizări: imagine curată, perturbația amplificată, imaginea atacată. Un
   README și un singur script care regenerează toate tabelele și figurile.

### Pe SVD imobiliare

1. **până vineri 11** Repo-ul Mariei e public:
   [github.com/mariatimbus/NYC\_RealEstate](https://github.com/mariatimbus/NYC_RealEstate),
   datele sunt cele de pe [Kaggle](https://www.kaggle.com/datasets/new-york-city/nyc-property-sales).
   Îl clonezi, construiești matricea de caracteristici cu codul ei, și pe matricea aia rulezi tu
   două detectoare de anomalii care nu depind de SVD: K-Means (distanța la centroid, k = 4) și
   [Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).
   Scoți lista de proprietăți marcate de fiecare la percentila 95. Asta e livrabilul tău de vineri,
   nu ai nevoie de nimic de la Maria.
2. **până luni 14** Adaugi al treilea detector, eroarea de reconstrucție SVD, folosind SVD-ul
   Mariei, și scoți și lista lui. Acum ai trei liste pe aceleași date.
3. **până joi 17** Acordul dintre metode: Jaccard pe fiecare pereche, câte proprietăți sunt
   marcate de toate trei, câte doar de una. Sensibilitate: refaci la praguri 90, 95, 99 și la rang
   k = 3, 5, 10.
4. **până duminică 20** Robustețe: scoți top 1% cele mai scumpe tranzacții și refaci; schimbi
   standardizarea cu una robustă (mediană și IQR) și refaci. Dacă listele rămân în mare aceleași,
   rezultatul e solid. Pachet reproductibil, ca la Attack Zoo.
5. **20 - 25 sep** Scrii secțiunea de transferabilitate la Attack Zoo și secțiunea de anomalii
   și robustețe la SVD.

## Întâlnirile

| Când | Ce vreau să văd |
|---|---|
| **Vineri 11 sep, 20:00** | Primul tău pas făcut și rulând. Trimitem abstractele. |
| **Luni 14 sep, seara** | Al doilea pas. De aici task-urile se leagă cu ale colegilor. |
| **Joi 17 sep, seara** | Grosul experimentelor. |
| **Duminică 20 sep** | Experimentele înghețate. Toate tabelele și figurile în repo. După ziua asta nu mai atingem experimentele. |
| **20 - 25 sep** | Scrii secțiunea ta în `paper/` din repo, ca markdown, direct din `results/`. Miercuri 24 ne vedem pe draft. |
| **25 - 27 sep** | Alex face polish și încarcă. |

## Când ai nevoie de mine

La întâlnirile de mai sus și atât. Dacă te blochezi între ele, scrii în canal unde te-ai oprit și treci la
următorul pas din listă; nu stai pe loc așteptând răspuns. Regula de „gata": un pas e gata când există un
script care îl rulează de la zero și un CSV sau o figură comisă în repo.

---
*Grup de cercetare MLSP · mentor: Alex Deonise · coordonator: Răzvan Rughiniș · RoEduNet 2026*
