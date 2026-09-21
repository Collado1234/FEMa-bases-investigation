#!/usr/bin/env python3
"""
gerar_ranking_femA.py

Recria, a partir do CSV `ranking_pairwise` (saída da análise `statistical_comparison`
com Friedman + post-hoc Wilcoxon + correção de Holm), duas coisas:

1. Tabelas-resumo (CSV/XLSX) com, para cada dataset x base FEMa:
   - quantas das outras bases ela superou (nominal, pelo sinal de mean_diff_1_minus_2)
   - quantas dessas vitórias são estatisticamente significativas após Holm
   - quantas derrotas significativas ela sofreu
   - sua posição no ranking daquele dataset

2. Um dashboard HTML autocontido e interativo (ranking + matriz de significância
   por dataset), igual ao publicado no chat.

USO
---
    python gerar_ranking_femA.py --input ranking_pairwise.csv --output-dir results/statistical_visualization

    # se as colunas numéricas da planilha vierem corrompidas (ver observação abaixo),
    # nada muda: o script já ignora magnitude e usa só sinal + flag de significância.

ENTRADA ESPERADA
-----------------
CSV com pelo menos as colunas:
    dataset, metric, basis_1, basis_2, significant_after_holm, mean_diff_1_minus_2

Cada par (basis_1, basis_2) deve aparecer UMA vez por dataset/metric (não as duas direções).

OBSERVAÇÃO IMPORTANTE
----------------------
Este script NÃO refaz nenhum teste estatístico. Ele só reorganiza e visualiza o que já
está no `ranking_pairwise`. Se as colunas de magnitude (p_value, p_value_holm,
effect_size_rank_biserial_r, mean_diff_1_minus_2) estiverem corrompidas por algum
problema de exportação de planilha (ex.: valores tipo "16.431.212.425.231.900" em vez de
p-values plausíveis), o script continua funcionando normalmente, pois usa apenas:
  - o SINAL de mean_diff_1_minus_2 (para saber qual base teve média maior); e
  - o BOOLEANO significant_after_holm (para saber se a diferença é confiável).
Esses dois campos não dependem da magnitude e continuam interpretáveis mesmo corrompidos
os demais. Ainda assim, se puder, corrija a exportação na fonte para recuperar p-values
e effect sizes reais — eles não são reconstruídos aqui.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLS = [
    "dataset", "metric", "basis_1", "basis_2",
    "significant_after_holm", "mean_diff_1_minus_2",
]


def sign(x) -> int:
    """Extrai só o sinal de um valor numérico possivelmente corrompido em formatação."""
    s = str(x).strip()
    return -1 if s.startswith("-") else 1


def to_bool(x) -> bool:
    return str(x).strip().upper() == "TRUE"


def load_pairwise(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes no CSV: {missing}")
    df = df[REQUIRED_COLS].copy()
    df["significant_after_holm"] = df["significant_after_holm"].apply(to_bool)
    df["sign"] = df["mean_diff_1_minus_2"].apply(sign)
    df["winner"] = df.apply(lambda r: r.basis_1 if r["sign"] == 1 else r.basis_2, axis=1)
    df["loser"] = df.apply(lambda r: r.basis_2 if r["sign"] == 1 else r.basis_1, axis=1)
    return df


def build_summary(df: pd.DataFrame):
    """Retorna (long_df, viz_data) — tabela tidy e estrutura para o dashboard HTML."""
    rows = []
    viz_data = {}

    for (dataset, metric), g in df.groupby(["dataset", "metric"]):
        bases = sorted(set(g.basis_1) | set(g.basis_2))
        wins = {b: 0 for b in bases}
        sig_wins = {b: 0 for b in bases}
        sig_losses = {b: 0 for b in bases}
        matrix = {}

        for _, r in g.iterrows():
            wins[r.winner] += 1
            if r.significant_after_holm:
                sig_wins[r.winner] += 1
                sig_losses[r.loser] += 1
            matrix[f"{r.basis_1}|{r.basis_2}"] = {
                "winner": r.winner,
                "sig": bool(r.significant_after_holm),
            }

        order = sorted(bases, key=lambda b: (-wins[b], -sig_wins[b], b))
        rank = {b: i + 1 for i, b in enumerate(order)}

        for b in bases:
            rows.append({
                "dataset": dataset,
                "metric": metric,
                "basis": b,
                "n_comparacoes": len(bases) - 1,
                "vitorias_totais": wins[b],
                "vitorias_significativas": sig_wins[b],
                "vitorias_nominais": wins[b] - sig_wins[b],
                "derrotas_significativas": sig_losses[b],
                "rank": rank[b],
            })

        viz_data.setdefault(dataset, {})[metric] = {
            "order": order, "wins": wins, "sig_wins": sig_wins,
            "sig_losses": sig_losses, "rank": rank, "matrix": matrix,
            "n_bases": len(bases),
        }

    long_df = pd.DataFrame(rows).sort_values(["dataset", "rank"]).reset_index(drop=True)
    return long_df, viz_data


def write_tables(long_df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    long_path = output_dir / "resumo_vitorias_por_dataset_basis.csv"
    long_df.to_csv(long_path, index=False)

    # tabelas largas: linha = dataset, coluna = basis
    wide_wins = long_df.pivot(index="dataset", columns="basis", values="vitorias_totais")
    wide_sig = long_df.pivot(index="dataset", columns="basis", values="vitorias_significativas")
    wide_loss = long_df.pivot(index="dataset", columns="basis", values="derrotas_significativas")

    xlsx_path = output_dir / "tabela_dataset_x_basis.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        long_df.to_excel(writer, sheet_name="long_tidy", index=False)
        wide_wins.to_excel(writer, sheet_name="vitorias_totais")
        wide_sig.to_excel(writer, sheet_name="vitorias_significativas")
        wide_loss.to_excel(writer, sheet_name="derrotas_significativas")

    # também por dataset individual, se quiser inspecionar isoladamente
    for dataset, g in long_df.groupby("dataset"):
        (output_dir / dataset).mkdir(exist_ok=True)
        g.sort_values("rank").to_csv(output_dir / dataset / f"summary_{g.metric.iloc[0]}.csv", index=False)

    return long_path, xlsx_path


def write_dashboard(viz_data: dict, output_dir: Path, template_path: Path = None):
    """Gera o dashboard HTML autocontido. Reusa o template embutido abaixo."""
    datasets = sorted(viz_data.keys())
    html = _HTML_TEMPLATE.replace(
        "__DATA_JSON__", json.dumps(viz_data, ensure_ascii=False)
    ).replace(
        "__DATASETS_JSON__", json.dumps(datasets, ensure_ascii=False)
    )
    out_path = output_dir / "dashboard_ranking_significancia.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="UTF-8">
<title>Comparação de bases FEMa</title>
<style>
body{font-family:sans-serif;margin:0;background:#eae6dc;color:#20241f;}
.layout{display:flex;min-height:100vh;}
.rail{width:230px;border-right:1px solid #d9d2c0;padding:20px;overflow-y:auto;}
.rail button{display:block;width:100%;text-align:left;background:none;border:none;
  font-family:monospace;font-size:0.82rem;padding:7px 8px;border-radius:5px;cursor:pointer;color:#726b5c;}
.rail button.active{background:#2f6f5e;color:#fff;}
.main{flex:1;padding:30px 40px;max-width:960px;}
h1{font-size:1.6rem;margin:0 0 10px 0;}
.rank-row{display:grid;grid-template-columns:24px 160px 1fr 140px;gap:10px;align-items:center;padding:4px 0;font-family:monospace;font-size:0.8rem;}
.rank-bar-track{background:#e6e1d3;border-radius:3px;height:14px;overflow:hidden;display:flex;}
.rank-bar-sig{background:#2f6f5e;height:100%;}
.rank-bar-ns{background:#c7ddd5;height:100%;}
table{border-collapse:collapse;font-family:monospace;font-size:0.62rem;margin-top:20px;}
td,th{width:22px;height:22px;text-align:center;border:1px solid #eae6dc;padding:0;}
.cell-diag{background:#d9d2c0;}
.cell-sig-win{background:#2f6f5e;}
.cell-sig-lose{background:#a6432e;}
.cell-ns{background:#e6e1d3;color:#726b5c;}
th.rowhead{text-align:right;padding-right:6px;border:none;font-size:0.65rem;color:#726b5c;}
th.colhead{writing-mode:vertical-rl;transform:rotate(180deg);border:none;font-size:0.62rem;color:#726b5c;}
.caveat{background:#fffdf8;border-left:3px solid #b9832c;padding:12px 14px;font-size:0.82rem;color:#726b5c;margin-bottom:24px;max-width:640px;}
</style></head>
<body>
<div class="layout">
  <aside class="rail"><h2>Datasets</h2><div id="list"></div></aside>
  <main class="main">
    <h1 id="title"></h1>
    <div class="caveat">Ranking construído a partir do sinal de mean_diff e de significant_after_holm
    (Wilcoxon + Holm), sem depender da magnitude bruta da planilha.</div>
    <div id="rankList"></div>
    <table id="matrix"></table>
  </main>
</div>
<script>
const DATA = __DATA_JSON__;
const DATASETS = __DATASETS_JSON__;
let current = DATASETS[0];

function render(){
  document.getElementById('list').innerHTML = DATASETS.map(ds =>
    `<button class="${ds===current?'active':''}" onclick="current='${ds}';render()">${ds}</button>`
  ).join('');
  const metric = Object.keys(DATA[current])[0];
  const d = DATA[current][metric];
  document.getElementById('title').textContent = current + ' (' + metric + ')';
  const maxWins = d.n_bases - 1;
  document.getElementById('rankList').innerHTML = d.order.map((b,i) => {
    const w = d.wins[b], sw = d.sig_wins[b], nw = w - sw;
    return `<div class="rank-row"><div>${i+1}</div><div>${b}</div>
      <div class="rank-bar-track">
        <div class="rank-bar-sig" style="width:${sw/maxWins*100}%"></div>
        <div class="rank-bar-ns" style="width:${nw/maxWins*100}%"></div>
      </div>
      <div>${sw} sig / ${nw} nom</div></div>`;
  }).join('');

  const order = d.order;
  let html = '<tr><th></th>' + order.map(b=>`<th class="colhead">${b}</th>`).join('') + '</tr>';
  order.forEach(rb => {
    html += `<tr><th class="rowhead">${rb}</th>`;
    order.forEach(cb => {
      if(rb===cb){ html += '<td class="cell-diag"></td>'; return; }
      const e = d.matrix[rb+'|'+cb] || d.matrix[cb+'|'+rb];
      const rowWins = e.winner === rb;
      if(e.sig){ html += `<td class="${rowWins?'cell-sig-win':'cell-sig-lose'}"></td>`; }
      else { html += `<td class="cell-ns">${rowWins?'+':'\u2013'}</td>`; }
    });
    html += '</tr>';
  });
  document.getElementById('matrix').innerHTML = html;
}
render();
</script>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Caminho do CSV ranking_pairwise")
    ap.add_argument("--output-dir", default="results/statistical_visualization",
                     help="Diretório de saída (default: results/statistical_visualization)")
    args = ap.parse_args()

    csv_path = Path(args.input)
    output_dir = Path(args.output_dir)

    df = load_pairwise(csv_path)
    long_df, viz_data = build_summary(df)
    long_path, xlsx_path = write_tables(long_df, output_dir)
    dashboard_path = write_dashboard(viz_data, output_dir)

    print(f"OK. {df['dataset'].nunique()} datasets, {len(df)} comparações pareadas.")
    print(f"- Tabela tidy (CSV): {long_path}")
    print(f"- Tabela dataset x basis (XLSX): {xlsx_path}")
    print(f"- Dashboard HTML: {dashboard_path}")


if __name__ == "__main__":
    main()