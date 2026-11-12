# Comparacao estatistica das bases (balanced_accuracy; fonte: holdout de teste; seeds: todas)

- Datasets na matriz: **28** (usados no teste global: 28; 0 sem informacao porque todas as bases empataram: nenhum)
- Bases: **21** | controle: **shepard** | alpha = 0.05
- Seeds por dataset (interseccao entre bases): min 1, max 3
- Datasets excluidos por dados incompletos: nenhum

## Teste global (Friedman + Iman-Davenport, Demsar 2006)

- chi2_F = 65.595 (p = 9.38e-07)
- F_F = 3.582, gl = (20, 540), **p = 3.34e-07**

Distancia critica de Nemenyi (so' para o diagrama): **CD = 5.92** postos.

## Ranking por posto medio

| basis                  |   posto_medio |   media_metrica |
|:-----------------------|--------------:|----------------:|
| radial                 |        7.5357 |          0.856  |
| rational_quadratic     |        8.75   |          0.8545 |
| gen_exponential        |        8.7857 |          0.8524 |
| lorentzian             |        9.1964 |          0.8535 |
| sigmoidal              |        9.1964 |          0.8527 |
| cauchy                 |        9.25   |          0.8507 |
| softmax_radial         |        9.375  |          0.8509 |
| laplacian              |        9.375  |          0.8509 |
| shepard                |       10.3571 |          0.8475 |
| rbf_gaussian           |       10.9821 |          0.8488 |
| attention              |       11.0357 |          0.8477 |
| entropic               |       11.0357 |          0.8493 |
| cubic_spline           |       11.9286 |          0.8488 |
| student_t              |       12.0714 |          0.8464 |
| inverse_multiquadratic |       12.1607 |          0.8458 |
| logarithmic            |       12.25   |          0.8459 |
| quartic_spline         |       12.4821 |          0.8482 |
| harmonic               |       12.8393 |          0.8434 |
| wendland_c2            |       12.8571 |          0.8474 |
| cosine                 |       13.2679 |          0.847  |
| multiquadratic         |       16.2679 |          0.8334 |

## Pos-hoc: cada base vs controle `shepard` (Wilcoxon signed-rank, zeros: Pratt; Holm)

| basis                  |      p |    r_rb |   wins |   ties |   losses |   mean_diff |   p_holm | significativo_holm_0.05   |
|:-----------------------|-------:|--------:|-------:|-------:|---------:|------------:|---------:|:--------------------------|
| multiquadratic         | 0.0058 | -0.5933 |      4 |      4 |       20 |     -0.0142 |   0.1153 | False                     |
| harmonic               | 0.0166 | -0.5667 |      7 |      4 |       17 |     -0.0041 |   0.3152 | False                     |
| radial                 | 0.1114 |  0.4071 |     14 |      6 |        8 |      0.0084 |   1      | False                     |
| rational_quadratic     | 0.1202 |  0.3659 |     15 |      5 |        8 |      0.007  |   1      | False                     |
| attention              | 0.2492 | -0.2567 |      9 |      4 |       15 |      0.0001 |   1      | False                     |
| sigmoidal              | 0.2595 |  0.2593 |     15 |      2 |       11 |      0.0052 |   1      | False                     |
| gen_exponential        | 0.2939 |  0.2267 |     15 |      4 |        9 |      0.0049 |   1      | False                     |
| laplacian              | 0.3789 |  0.2246 |     13 |      5 |       10 |      0.0034 |   1      | False                     |
| softmax_radial         | 0.3789 |  0.2246 |     13 |      5 |       10 |      0.0034 |   1      | False                     |
| cosine                 | 0.3931 | -0.1799 |     10 |      1 |       17 |     -0.0005 |   1      | False                     |
| inverse_multiquadratic | 0.4227 | -0.1344 |      8 |      6 |       14 |     -0.0017 |   1      | False                     |
| wendland_c2            | 0.5162 | -0.1282 |     10 |      2 |       16 |     -0.0002 |   1      | False                     |
| logarithmic            | 0.553  | -0.1467 |     11 |      4 |       13 |     -0.0016 |   1      | False                     |
| student_t              | 0.6897 | -0.09   |     11 |      4 |       13 |     -0.0011 |   1      | False                     |
| lorentzian             | 0.7583 |  0.0738 |     13 |      3 |       12 |      0.006  |   1      | False                     |
| cauchy                 | 0.8093 |  0      |      9 |      7 |       12 |      0.0032 |   1      | False                     |
| quartic_spline         | 0.8823 | -0.0142 |     11 |      2 |       15 |      0.0007 |   1      | False                     |
| cubic_spline           | 0.9184 |  0.0317 |     12 |      1 |       15 |      0.0013 |   1      | False                     |
| rbf_gaussian           | 0.9274 |  0.0154 |     13 |      3 |       12 |      0.0013 |   1      | False                     |
| entropic               | 0.9365 | -0.0085 |     12 |      2 |       14 |      0.0018 |   1      | False                     |

## Baselines (Wilcoxon pareado entre datasets, Holm)

| base    | baseline   |   n_datasets |      p |    r_rb |   wins |   ties |   losses |   mean_diff |   p_holm |
|:--------|:-----------|-------------:|-------:|--------:|-------:|-------:|---------:|------------:|---------:|
| shepard | knn        |           28 | 0.5922 | -0.1169 |     11 |      3 |       14 |     -0.0018 |        1 |
| shepard | logreg     |           28 | 0.438  | -0.1724 |     12 |      0 |       16 |      0.0149 |        1 |
| radial  | knn        |           28 | 0.2836 |  0.2333 |     15 |      4 |        9 |      0.0066 |        1 |
| radial  | logreg     |           28 | 0.8314 |  0.0493 |     14 |      0 |       14 |      0.0234 |        1 |

(`radial` e' a base de melhor posto medio: escolhida a posteriori, interprete com cautela.)


*Unidade de analise: dataset. Cada celula e' a media da metrica de teste nas seeds comuns.*