# Bateria estatística para a comparação de funções de base (FEMa) — metodologia e justificativas

## 0. Perguntas → camadas

| Pergunta | Camada | Unidade de análise | Testes |
|---|---|---|---|
| (A) Em **cada dataset**, quais bases são melhores / piores / equivalentes a quais? | `per_dataset.py` | bloco pareado (seed × repeat × fold) | Friedman+Iman-Davenport (omnibus), t pareado com variância corrigida (Nadeau-Bengio), TOST, Wilcoxon (robustez), KS/CvM/AD pareados (forma da distribuição), GoF de normalidade (diagnóstico) |
| (B) O desempenho relativo **depende do contexto** (nº amostras, features, classes, desbalanceamento, IHDD)? | `by_group.py` | dataset | Friedman por grupo, teste de permutação global do perfil de postos × grupo, Kruskal-Wallis por base, Spearman posto × covariável; tudo também ao nível de **família** de bases |
| (C) A diferença persiste na **qualidade de ranqueamento** (AUC) no holdout, sem viés de seleção? | `delong.py` | amostras do conjunto de teste | DeLong (AUCs correlacionadas) |

## 1. Por que os dados por fold exigem cuidado

1. **Pareamento.** Os splits dependem só de `(master_seed, estratégia)` e dos dados (`tuning/cv_strategy.py`), logo são idênticos entre bases: o fold *i* da base A e da base B avaliam as mesmas amostras. A diferença `d_i = A_i − B_i` cancela a dificuldade do fold. Todos os testes por dataset são pareados por `(seed, repeat, fold)`; blocos incompletos são descartados (`loading.complete_blocks`).
2. **Não independência.** Os conjuntos de treino de folds distintos se sobrepõem (~80–90 %), então os 30 valores *não* são 30 observações independentes. Nadeau & Bengio (2003) mostram que `Var(d̄) = s²·(1/n + ρ)`, com `ρ = n_val/n_train`. É o t pareado "corrigido" (`primitives.corrected_t`). Com 10 folds ρ≈0,11 e `n_eff = 1/(1/n+ρ) ≈ 7`.
3. **Viés de seleção.** O escore por fold é da configuração **vencedora de 20** no mesmo CV (winner's curse). Isso favorece igualmente todas as bases *em média*, mas bases com espaço de busca mais sensível ganham mais "sorte". O relatório traz o diagnóstico `melhor − mediana` das combinações por base. O **holdout** (camada C) é o árbitro não enviesado.

## 2. Camada A — decisões de projeto

- **Omnibus:** Friedman + Iman-Davenport (Demšar 2006) sobre os blocos, em duas versões: ingênua e com `N → N/deff` (`deff = 1 + nρ`). A ingênua é anticonservadora e fica só como referência.
- **Pares (todas as C(k,2)):** t pareado corrigido → `p`; Holm dentro do dataset (FWER) para **afirmar diferença**.
- **Equivalência ("igual a tanto").** `p > 0,05` **não** prova igualdade. Usa-se TOST (Schuirmann 1987) com o mesmo erro-padrão corrigido e margem `δ` (padrão 0,01 = 1 p.p. da métrica; **defina antes de ver os resultados**). Família de multiplicidade própria: BH (FDR) por padrão, porque Holm sobre 210 pares torna equivalência inalcançável (verificado no sintético); `--equiv-adjust holm|none` para sensibilidade.
- **Desfecho de 3 vias:** `equivalente` (TOST) › `a_melhor`/`b_melhor` (t corrigido + Holm) › `inconclusivo`.
- **Dois níveis de evidência por par:** *robusto* (sobrevive à correção N-B + Holm) e *indicativo* (só sem a correção, mas com Holm). Existe porque a calibração (§5) mostrou que N-B é muito conservador em cenários de aprendiz estável. Conclusões fortes devem usar só *robusto*.
- **KS, Cramér-von Mises, Anderson-Darling (EDF).** Testam se a *distribuição inteira* dos escores difere (posição, dispersão, assimetria) — não "quem é melhor". Usos: (i) achar pares **equivalentes em posição mas com forma diferente** (ex.: mesma média, estabilidade diferente) → `equiv_mas_forma_difere`; (ii) conferir que o t não esconde caudas. Implementados com p-valor de **permutação pareada** (troca A↔B dentro do bloco), tie-aware; estatísticas conferem com `scipy` (`tests/`). A versão não pareada foi **descartada**: sob pareamento ela é conservadora demais (0 % de falsos positivos e poder ≈ 0 na calibração). `forma_difere` exige ≥ 2 dos 3 testes (Holm) rejeitarem.
- **Aderência à Normal** (KS-Lilliefors, CvM, AD com parâmetros estimados; p por Monte Carlo): coluna `normal_ok` para `d_i`. É diagnóstico — não decide nada; serve para ler o t com cautela quando `normal_ok=False`.
- **Grupo do topo:** bases não significativamente piores que a melhor em média (decisão `robusto`, e uma versão `indicativo`). Mede "quantas bases são indistinguíveis da melhor" — mais informativo que o vencedor por média, que em muitos datasets é ruído.

## 3. Camada B — por grupo

- **Unidade = dataset** (Demšar): escore do dataset = média dos blocos. Não há pseudo-replicação por folds.
- **Dentro de cada grupo:** Friedman + Iman-Davenport e distância crítica de Nemenyi. Não se usa Wilcoxon+Holm em grupos pequenos: com n=7 o menor p exato é 2/2⁷ = 0,0156 > 0,05/20, ou seja, **poder nulo** por construção.
- **O grupo importa?** Teste de permutação global: `T = Σ_g n_g‖r̄_g − r̄‖²` sobre os vetores de posto médio; rótulos permutados entre datasets. Sem hipótese distribucional, válido com n pequeno, uma decisão por critério (Holm entre os 5 critérios).
- **Qual base muda?** Kruskal-Wallis por base (Holm entre bases×critérios) e Spearman posto × log(amostras, features, desbalanceamento) e nº de classes (BH). Só interpretar após o teste global.
- **Famílias** (`loading.FAMILIES`, por comportamento de decaimento de φ(d)): potência/algébrica singular; cauda pesada; exponencial/gaussiana; suporte compacto; não-decrescente/sigmoide. Escore da família = média das bases membros (não "melhor da família", que favoreceria famílias grandes). Reduz 21 → 5 níveis e aumenta o poder. **Confira o agrupamento** — é uma hipótese de modelagem sua.

## 4. Camada C — DeLong

Compara AUCs correlacionadas (mesmas amostras) sem pressupor independência entre bases (DeLong et al. 1988; Sun & Xu 2014, vetorizado). Binário: AUC direta. Multiclasse: um-contra-resto por classe, Holm entre classes (H1: difere em ≥ 1 classe). **Requer predições por amostra, que `results/` não guarda**: `scripts/dump_test_predictions.py` refaz o *final fit* (mesma seed do pipeline), salva as probabilidades do holdout e confere o `roc_auc` contra `test_results.json`. Cada seed tem outro holdout; os resultados são por (dataset, seed) e a conclusão exige consistência entre seeds. Mede discriminação (AUC), não acurácia balanceada: pode divergir do restante e isso é informativo.

## 5. Calibração empírica (`tests/calibration_null.py`)

Regressão logística, n=300, CV 10×3 real; A e B treinam em subamostras aleatórias independentes do treino de cada fold. H0: ambos 80 %; H1: B com 50 % (efeito pequeno). 300 experimentos cada (erro-padrão da taxa ≈ 1,3 p.p.).

| teste | falso positivo (H0) | poder (H1) |
|---|---|---|
| t ingênuo | 6,3 % | 14,0 % |
| t corrigido (N-B) | 0,0 % | 0,0 % |
| Wilcoxon | 8,0 % | 11,7 % |
| KS não pareado | 0,0 % | 0,3 % |
| KS / CvM / AD pareado | 2,3 % / 6,0 % / 3,3 % | 5,7 % / 11,3 % / 12,0 % |
| KS / CvM / AD pareado + deff | 0 % / 0 % / 0 % | 0 % / 0 % / 0 % |

Leitura honesta: neste cenário o t ingênuo está próximo do nominal; N-B é conservador demais (sem poder); o pareamento é essencial para EDF. **Efeitos de ~1 p.p. com 30 folds são praticamente indetectáveis com qualquer correção** — espere muitos `inconclusivo`. O cenário usa aprendiz estável com diferença aleatória; bases com diferença *sistemática* tendem a ter mais dependência entre folds, que este teste não captura.

## 6. Ameaças à validade (leia antes de concluir)

1. **Poder.** Grupos com 3–7 datasets quase não detectam nada; "não significativo" ≠ "sem efeito".
2. **Multiplicidade:** até 21·20/2 × 28 ≈ 5 900 testes por métrica. Afirmações por dataset usam Holm; o conjunto de datasets é tratado como exploratório (BH no omnibus).
3. **Não independência entre datasets:** `mfeat_*` são 5 representações dos mesmos 2000 objetos. Faça análise de sensibilidade colapsando-os (feito na camada B real: os sinais nominais mudaram de lugar).
4. **Métrica e margem `δ`** devem ser fixadas antes dos resultados. O v3 usa `balanced_accuracy` para análise enquanto o tuning otimiza `f1`; mantenha isso explícito.
5. **Seeds** têm splits treino/teste diferentes; os blocos são agrupados com ρ constante (conservador), mas seeds não são independentes de fato.
6. **Exploratório ≠ confirmatório.** Padrões encontrados em subgrupos devem ser confirmados em datasets novos ou por pré-registro.

## 7. Como rodar

```bash
# unzip na raiz do projeto; pip install tabulate
python -m reporting.stat_battery.cli --results results --prefix v3 --metric balanced_accuracy \
    --groups reports/eda/dataset_characterization.csv --out reports/v4_stats --delta 0.01
python scripts/dump_test_predictions.py --prefix v3 --out preds/      # opcional (camada C)
python -m reporting.stat_battery.cli ... --preds preds/
python tests/test_stat_battery.py          # 7 testes (sem pytest)
python tests/calibration_null.py --reps 300
```
Custo: a 1ª execução lê ~10⁵–10⁶ JSONs (cache em `<out>/cache_*.csv`); a camada A usa 999 permutações por par (~1 min para 28 datasets × 210 pares em 1 CPU, ajustável com `--n-perm`).

Referências: Demšar 2006 (JMLR 7); Nadeau & Bengio 2003 (Mach. Learn. 52); Schuirmann 1987; DeLong, DeLong & Clarke-Pearson 1988 (Biometrics 44); Sun & Xu 2014 (IEEE SPL 21); Scholz & Stephens 1987 (JASA 82); Benavoli et al. 2017 (JMLR 18); Bouckaert & Frank 2004.
