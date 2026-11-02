#!/usr/bin/env python3
"""
gerar_visao_geral_femA.py

Constroi uma VISAO GERAL das bases FEMa (agregando todos os datasets) e cruza com a
categorizacao dos datasets (tamanho, classes, desbalanceamento, dificuldade IHDD).

Le a planilha mestre exportada do projeto, com (pelo menos) estas abas:
  - "Ranking": dataset, metric, posicao, basis            (posicao = 1..21, 1 = melhor;
        vem de um ranking ja calculado antes da corrupcao numerica, entao e confiavel)
  - "Datasets": Nome - DataSet, cat-feat, cat-samples, cat-classes, cat-ind, Cat-IHDD
        (categorizacao dos datasets)
  - "long_tidy" (opcional): dataset, basis, vitorias_significativas, derrotas_significativas,
        vitorias_totais  -- saida do gerar_ranking_femA.py, usada so para somar o total
        de vitorias/derrotas estatisticamente significativas por base. Se a aba nao
        existir, essas colunas saem zeradas e um aviso e impresso.

NAO recalcula nenhum teste estatistico; so agrega o que ja existe.

USO
---
    python gerar_visao_geral_femA.py --input "Classificacao_-_Datasets_usados.xlsx" \
        --output-dir results/statistical_visualization/visao_geral
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

CAT_COLS = ["cat-feat", "cat-samples", "cat-classes", "cat-ind", "Cat-IHDD"]
CAT_LEVEL_ORDER = {
    "cat-feat": ["Pequena", "Media", "Grande"],
    "cat-samples": ["Pequena", "Media", "Grande"],
    "cat-classes": ["Pequena", "Media", "Grande"],
    "cat-ind": ["Balanceado", "Leve", "Moderado", "Severo"],
    "Cat-IHDD": ["Baixa", "Moderada", "Alta", "Muito Alta"],
}
CAT_LABELS_PT = {
    "cat-feat": "Nº de features",
    "cat-samples": "Nº de amostras",
    "cat-classes": "Nº de classes",
    "cat-ind": "Desbalanceamento",
    "Cat-IHDD": "Dificuldade (IHDD)",
}


def clean_name(x) -> str:
    if not isinstance(x, str):
        return x
    return re.sub(r"[`'\"]", "", x).strip()


def load_ranking(xls: pd.ExcelFile) -> pd.DataFrame:
    df = pd.read_excel(xls, "Ranking")[["dataset", "metric", "posicao", "basis"]]
    df = df.dropna(subset=["dataset", "basis"])
    df["posicao"] = df["posicao"].astype(int)
    df["dataset"] = df["dataset"].apply(clean_name)
    return df


def load_dataset_categories(xls: pd.ExcelFile) -> pd.DataFrame:
    df = pd.read_excel(xls, "Datasets")
    df.columns = [str(c).strip() for c in df.columns]
    df["dataset"] = df["Nome - DataSet"].apply(clean_name)
    df = df.dropna(subset=["dataset"])
    keep = ["dataset"] + [c for c in CAT_COLS if c in df.columns]
    return df[keep].drop_duplicates("dataset")


def load_sig_counts(xls: pd.ExcelFile) -> pd.DataFrame:
    if "long_tidy" not in xls.sheet_names:
        print("Aviso: aba 'long_tidy' não encontrada — vitórias/derrotas significativas "
              "sairão zeradas. Rode primeiro gerar_ranking_femA.py e cole a aba 'long_tidy' "
              "na planilha, se quiser esses totais.")
        return pd.DataFrame(columns=["basis", "total_vitorias_sig", "total_derrotas_sig",
                                      "total_vitorias_totais"])
    lt = pd.read_excel(xls, "long_tidy")
    g = lt.groupby("basis")[["vitorias_significativas", "derrotas_significativas",
                              "vitorias_totais"]].sum()
    g.columns = ["total_vitorias_sig", "total_derrotas_sig", "total_vitorias_totais"]
    return g.reset_index()


def build_overview(ranking: pd.DataFrame, sig_counts: pd.DataFrame) -> pd.DataFrame:
    n_bases = ranking["basis"].nunique()
    grp = ranking.groupby("basis")["posicao"]
    overview = pd.DataFrame({
        "avg_posicao": grp.mean().round(2),
        "mediana_posicao": grp.median(),
        "std_posicao": grp.std().round(2),
        "melhor_posicao": grp.min(),
        "pior_posicao": grp.max(),
        "n_datasets": grp.count(),
        "n_top1": ranking[ranking.posicao == 1].groupby("basis").size(),
        "n_top3": ranking[ranking.posicao <= 3].groupby("basis").size(),
        "n_bottom3": ranking[ranking.posicao >= n_bases - 2].groupby("basis").size(),
    }).fillna(0)
    for c in ["n_top1", "n_top3", "n_bottom3", "n_datasets"]:
        overview[c] = overview[c].astype(int)

    overview = overview.join(sig_counts.set_index("basis"), how="left").fillna(0)
    for c in ["total_vitorias_sig", "total_derrotas_sig", "total_vitorias_totais"]:
        overview[c] = overview[c].astype(int)

    overview = overview.sort_values("avg_posicao")
    overview.insert(0, "rank_geral", range(1, len(overview) + 1))
    return overview.reset_index().rename(columns={"index": "basis"})


def build_category_tables(ranking: pd.DataFrame, cats: pd.DataFrame, overview: pd.DataFrame):
    merged = ranking.merge(cats, on="dataset", how="left")
    order = overview.set_index("basis")["rank_geral"].to_dict()
    tables = {}
    for col in CAT_COLS:
        if col not in merged.columns:
            continue
        pivot = merged.pivot_table(index="basis", columns=col, values="posicao", aggfunc="mean").round(2)
        levels = [l for l in CAT_LEVEL_ORDER[col] if l in pivot.columns]
        levels += [l for l in pivot.columns if l not in levels]
        pivot = pivot[levels]
        pivot = pivot.loc[sorted(pivot.index, key=lambda b: order.get(b, 999))]
        tables[col] = pivot
    return tables


def write_outputs(overview: pd.DataFrame, cat_tables: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    overview.to_csv(output_dir / "overview_geral_basis.csv", index=False)

    try:
        with pd.ExcelWriter(output_dir / "visao_geral_por_categoria.xlsx", engine="openpyxl") as writer:
            overview.to_excel(writer, sheet_name="overview_geral", index=False)
            for col, pivot in cat_tables.items():
                sheet = re.sub(r"[^A-Za-z0-9]", "_", col)[:31]
                pivot.to_excel(writer, sheet_name=sheet)
    except ImportError:
        print("Aviso: openpyxl não encontrado — pulando .xlsx (instale com "
              "'pip install openpyxl'). CSVs da visão geral já foram salvos.")
        for col, pivot in cat_tables.items():
            pivot.to_csv(output_dir / f"por_categoria_{re.sub(r'[^A-Za-z0-9]', '_', col)}.csv")


def write_dashboard(overview: pd.DataFrame, cat_tables: dict, output_dir: Path):
    overview_records = overview.to_dict(orient="records")
    cats_json = {
        col: {
            "label": CAT_LABELS_PT.get(col, col),
            "levels": list(pivot.columns),
            "data": {basis: {lvl: (None if pd.isna(v) else v) for lvl, v in row.items()}
                     for basis, row in pivot.iterrows()},
        }
        for col, pivot in cat_tables.items()
    }
    html = _HTML_TEMPLATE.replace("__OVERVIEW_JSON__", json.dumps(overview_records, ensure_ascii=False))
    html = html.replace("__CATS_JSON__", json.dumps(cats_json, ensure_ascii=False))
    out_path = output_dir / "dashboard_visao_geral.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="UTF-8">
<title>Visão geral — bases FEMa</title>
<style>
body{font-family:sans-serif;margin:0;background:#eae6dc;color:#20241f;}
.main{max-width:980px;margin:0 auto;padding:34px 30px 60px 30px;}
h1{font-size:1.6rem;margin:0 0 6px 0;}
.sub{color:#726b5c;font-size:0.85rem;margin-bottom:24px;}
.rank-row{display:grid;grid-template-columns:24px 170px 1fr 190px;gap:10px;align-items:center;padding:4px 0;font-family:monospace;font-size:0.8rem;}
.rank-bar-track{background:#e6e1d3;border-radius:3px;height:14px;position:relative;}
.rank-bar-range{background:#c7ddd5;height:100%;border-radius:3px;position:absolute;}
.rank-bar-mark{position:absolute;top:-3px;width:2px;height:20px;background:#2f6f5e;}
select{font-family:inherit;font-size:0.85rem;padding:6px 10px;border-radius:6px;border:1px solid #d9d2c0;background:#fffdf8;margin-bottom:14px;}
table{border-collapse:collapse;font-family:monospace;font-size:0.72rem;width:100%;}
td,th{padding:4px 8px;border:1px solid #d9d2c0;text-align:center;}
th{background:#fffdf8;}
td.basis-name{text-align:left;font-weight:600;}
.caveat{background:#fffdf8;border-left:3px solid #b9832c;padding:12px 14px;font-size:0.82rem;color:#726b5c;margin-bottom:24px;max-width:680px;}
section{margin-bottom:40px;}
h2{font-size:1.05rem;margin:0 0 10px 0;}
</style></head>
<body><div class="main">
<h1>Visão geral das bases FEMa</h1>
<div class="sub">Agregado de todos os datasets disponíveis, cruzado com a categorização dos datasets.</div>
<div class="caveat">O ranking geral usa a posição (1–21) já calculada por dataset, que não sofre a corrupção
numérica da planilha. As colunas de vitórias/derrotas significativas cobrem apenas os datasets em que a
comparação pareada (Wilcoxon+Holm) está disponível.</div>

<section>
  <h2>Ranking geral (média da posição em todos os datasets)</h2>
  <div id="rankList"></div>
</section>

<section>
  <h2>Desempenho por característica do dataset</h2>
  <select id="catSelect"></select>
  <table id="catTable"></table>
</section>
</div>
<script>
const OVERVIEW = __OVERVIEW_JSON__;
const CATS = __CATS_JSON__;

function renderRank(){
  const maxPos = Math.max(...OVERVIEW.map(r=>r.pior_posicao));
  document.getElementById('rankList').innerHTML = OVERVIEW.map(r => {
    const left = (r.melhor_posicao-1)/maxPos*100;
    const width = (r.pior_posicao-r.melhor_posicao)/maxPos*100;
    const mark = (r.avg_posicao-1)/maxPos*100;
    return `<div class="rank-row">
      <div>${r.rank_geral}</div>
      <div>${r.basis}</div>
      <div class="rank-bar-track">
        <div class="rank-bar-range" style="left:${left}%;width:${width}%"></div>
        <div class="rank-bar-mark" style="left:${mark}%"></div>
      </div>
      <div>méd ${r.avg_posicao} · ${r.total_vitorias_sig} vit.sig / ${r.total_derrotas_sig} der.sig</div>
    </div>`;
  }).join('');
}

function renderCatSelect(){
  const sel = document.getElementById('catSelect');
  sel.innerHTML = Object.keys(CATS).map(k => `<option value="${k}">${CATS[k].label}</option>`).join('');
  sel.onchange = renderCatTable;
}

function renderCatTable(){
  const key = document.getElementById('catSelect').value;
  const cat = CATS[key];
  const order = OVERVIEW.map(r=>r.basis);
  let html = '<tr><th>base</th>' + cat.levels.map(l=>`<th>${l}</th>`).join('') + '</tr>';
  order.forEach(basis => {
    const row = cat.data[basis] || {};
    html += `<tr><td class="basis-name">${basis}</td>` +
      cat.levels.map(l => `<td>${row[l] != null ? row[l] : '—'}</td>`).join('') + '</tr>';
  });
  document.getElementById('catTable').innerHTML = html;
}

renderRank();
renderCatSelect();
renderCatTable();
</script>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Planilha mestre (.xlsx) com abas Ranking e Datasets")
    ap.add_argument("--output-dir", default="results/statistical_visualization/visao_geral")
    args = ap.parse_args()

    xls = pd.ExcelFile(args.input)
    ranking = load_ranking(xls)
    cats = load_dataset_categories(xls)
    sig_counts = load_sig_counts(xls)

    overview = build_overview(ranking, sig_counts)
    cat_tables = build_category_tables(ranking, cats, overview)

    output_dir = Path(args.output_dir)
    write_outputs(overview, cat_tables, output_dir)
    dash_path = write_dashboard(overview, cat_tables, output_dir)

    print(f"OK. {ranking['dataset'].nunique()} datasets, {ranking['basis'].nunique()} bases.")
    print(f"- Overview geral: {output_dir}/overview_geral_basis.csv")
    print(f"- Tabelas por categoria: {output_dir}/visao_geral_por_categoria.xlsx")
    print(f"- Dashboard: {dash_path}")


if __name__ == "__main__":
    main()