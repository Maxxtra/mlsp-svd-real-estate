# Maria, planul tău până pe 27 septembrie

SVD imobiliare. Lista de mai jos e a ta: o iei de sus în jos, fiecare pas are termenul lui, care e întâlnirea la
care vreau să-l văd făcut. Primul pas e gândit să-l termini singur, fără să aștepți după nimeni.

*SVD imobiliare · nucleul SVD și modelarea prețului · repo: [github.com/Maxxtra/mlsp-svd-real-estate](https://github.com/Maxxtra/mlsp-svd-real-estate)*

1. **până vineri 11** Validarea SVD-ului tău față de `numpy.linalg.svd`, dar pe
   matricea reală de 29275×408, nu pe cea de test de 180×24: eroarea de reconstrucție, ortogonalitatea
   lui U și V, timpul de rulare. Scrii cifrele în Sheet. E complet independent de David.
2. **până luni 14** Aproximarea de rang redus: eroarea de reconstrucție pentru k de la 1 la 50
   și k-ul la care păstrezi 90% și 95% din informație. O figură, curba erorii în funcție de k, cu
   cele două praguri marcate. Tot de luni, David folosește SVD-ul tău pentru al treilea detector.
3. **până joi 17** Regresia: least-squares prin pseudo-inversă și Ridge cu 3-4 valori de α.
   Din reziduurile relative scoți top 50 subevaluate și top 50 supraevaluate, și te uiți dacă sunt
   concentrate în câteva cartiere sau împrăștiate.
4. **până duminică 20** Îngheți. Figurile finale: scree plot, heatmap-ul de loadings pe primele
   5 componente, distribuția reziduurilor cu pragurile marcate.
5. **20 - 25 sep** Scrii metoda (SVD-ul tău și regresia) și partea de mispricing.

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
