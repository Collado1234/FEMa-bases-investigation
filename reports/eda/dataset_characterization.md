# Caracterização dos datasets

Quantidade de datasets: 28.

Os metadados foram calculados diretamente a partir dos dados brutos, usando as especificações de `datasets/registry.py` e as rotinas de leitura de `datasets/loader.py`.

A contagem de features considera as colunas originais de entrada, após `drop_columns` e a remoção do target, mas antes de One-Hot Encoding.

## Quartis calculados nesta execução

| Dimensão | Q1 | Q3 |
|---|---:|---:|
| Features | 7.75 | 64 |
| Amostras | 552 | 2599.5 |
| Classes | 2 | 10 |

## Critérios

- Pequena: valor menor ou igual a Q1 (1 ponto).
- Média: valor acima de Q1 e menor ou igual a Q3 (2 pontos).
- Grande: valor acima de Q3 (3 pontos).
- Índice de desbalanceamento = contagem da classe majoritária / contagem da classe minoritária.
- Balanceado: índice até 1,5 (0 pontos); Leve: até 3 (1 ponto); Moderado: até 10 (2 pontos); Severo: acima de 10 (3 pontos).
- IHDD = pontos de features + amostras + classes + desbalanceamento.
- Categoria IHDD: até 5 = Moderada; 6–7 = Alta; 8 ou mais = Muito Alta.

O IHDD é uma heurística descritiva, não uma medida universal de dificuldade.

## Tabela

| dataset                          |   n_features |   n_amostras |   n_classes |   indice_desbalanceamento | cat_features   | cat_amostras   | cat_classes   | cat_desbalanceamento   |   IHDD | cat_IHDD   |
|:---------------------------------|-------------:|-------------:|------------:|--------------------------:|:---------------|:---------------|:--------------|:-----------------------|-------:|:-----------|
| blood_transfusion_service_center |            4 |          748 |           2 |                   3.20225 | Pequena        | Média          | Pequena       | Moderado               |      6 | Alta       |
| breast_cancer                    |           30 |          569 |           2 |                   1.68396 | Média          | Média          | Pequena       | Leve                   |      6 | Alta       |
| climate_model_simulation_crashes |           20 |          540 |           2 |                  10.7391  | Média          | Pequena        | Pequena       | Severo                 |      7 | Alta       |
| diabetes                         |            8 |          768 |           2 |                   1.86567 | Média          | Média          | Pequena       | Leve                   |      6 | Alta       |
| digits                           |           64 |         1797 |          10 |                   1.05172 | Média          | Média          | Média         | Balanceado             |      6 | Alta       |
| fetal_health                     |           21 |         2126 |           3 |                   9.40341 | Média          | Média          | Média         | Moderado               |      8 | Muito Alta |
| gina_agnostic                    |          970 |         3468 |           2 |                   1.03402 | Grande         | Grande         | Pequena       | Balanceado             |      7 | Alta       |
| haberman                         |            3 |          306 |           2 |                   2.77778 | Pequena        | Pequena        | Pequena       | Leve                   |      4 | Moderada   |
| iris                             |            4 |          150 |           3 |                   1       | Pequena        | Pequena        | Média         | Balanceado             |      4 | Moderada   |
| isolet                           |          617 |         7797 |          26 |                   1.00671 | Grande         | Grande         | Grande        | Balanceado             |      9 | Muito Alta |
| letter                           |           16 |        20000 |          26 |                   1.10763 | Média          | Grande         | Grande        | Balanceado             |      8 | Muito Alta |
| magic_telescope                  |           10 |        19020 |           2 |                   1.8439  | Média          | Grande         | Pequena       | Leve                   |      7 | Alta       |
| mfeat_factors                    |          216 |         2000 |          10 |                   1       | Grande         | Média          | Média         | Balanceado             |      7 | Alta       |
| mfeat_fourier                    |           76 |         2000 |          10 |                   1       | Grande         | Média          | Média         | Balanceado             |      7 | Alta       |
| mfeat_karhunen                   |           64 |         2000 |          10 |                   1       | Média          | Média          | Média         | Balanceado             |      6 | Alta       |
| mfeat_pixel                      |          240 |         2000 |          10 |                   1       | Grande         | Média          | Média         | Balanceado             |      7 | Alta       |
| mfeat_zernike                    |           47 |         2000 |          10 |                   1       | Média          | Média          | Média         | Balanceado             |      6 | Alta       |
| monks_problems_1                 |            6 |          556 |           2 |                   1       | Pequena        | Média          | Pequena       | Balanceado             |      4 | Moderada   |
| monks_problems_2                 |            6 |          601 |           2 |                   1.91748 | Pequena        | Média          | Pequena       | Leve                   |      5 | Moderada   |
| musk                             |          167 |         6598 |           2 |                   5.48771 | Grande         | Grande         | Pequena       | Moderado               |      9 | Muito Alta |
| page_blocks                      |           10 |         5473 |           5 |                 175.464   | Média          | Grande         | Média         | Severo                 |     10 | Muito Alta |
| seeds                            |            7 |          210 |           3 |                   1       | Pequena        | Pequena        | Média         | Balanceado             |      4 | Moderada   |
| segment                          |           19 |         2310 |           7 |                   1       | Média          | Média          | Média         | Balanceado             |      6 | Alta       |
| spambase                         |           57 |         4601 |           2 |                   1.53778 | Média          | Grande         | Pequena       | Leve                   |      7 | Alta       |
| tic_tac_toe                      |            9 |          958 |           2 |                   1.88554 | Média          | Média          | Pequena       | Leve                   |      6 | Alta       |
| vertebra_column                  |            6 |          310 |           3 |                   2.5     | Pequena        | Pequena        | Média         | Leve                   |      5 | Moderada   |
| wine                             |           13 |          178 |           3 |                   1.47917 | Média          | Pequena        | Média         | Balanceado             |      5 | Moderada   |
| zoo                              |           16 |          101 |           7 |                  10.25    | Média          | Pequena        | Média         | Severo                 |      8 | Muito Alta |
