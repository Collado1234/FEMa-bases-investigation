#!/usr/bin/env bash
# reorganize_project.sh
#
# Reorganiza o projeto FEMa-bases-investigation, ARQUIVANDO (nao apagando)
# codigo morto, redundante ou de scripts pontuais ja concluidos, para uma
# pasta archive/YYYY-MM-DD/ com a mesma estrutura de subpastas original --
# tudo reversivel (e' so' mover de volta se precisar de algo).
#
# Rode este script a partir da RAIZ do projeto:
#   bash reorganize_project.sh
#
# O que ele faz, em ordem:
#   1. Cria archive/<data-de-hoje>/
#   2. Move para la' os arquivos identificados na auditoria como:
#        - pipeline estatistica B (redundante e com risco de contradizer
#          analysis/statistical_comparison.py + analysis/ranking_report.py,
#          que sao a pipeline OFICIAL a partir de agora)
#        - codigo morto (src/, run_experiment.py, reporting/compare_models.py)
#        - scripts de diagnostico pontuais/scratch ja concluidos
#   3. Imprime um resumo do que foi movido (ou avisa se algum arquivo ja
#      nao existia, sem quebrar o script)
#
# NADA e' deletado. Se precisar de algum arquivo depois, esta' em
# archive/<data>/<caminho original>.

set -uo pipefail

TODAY=$(date +%Y-%m-%d)
ARCHIVE_DIR="archive/${TODAY}"

echo "Criando ${ARCHIVE_DIR}/ ..."
mkdir -p "${ARCHIVE_DIR}"

move_if_exists() {
    local src="$1"
    if [ -e "$src" ]; then
        local dest_dir
        dest_dir="${ARCHIVE_DIR}/$(dirname "$src")"
        mkdir -p "$dest_dir"
        mv "$src" "$dest_dir/"
        echo "  [movido] $src -> $dest_dir/"
    else
        echo "  [pulado, nao existe] $src"
    fi
}

echo ""
echo "=== 1. Pipeline estatistica B (redundante -- analysis/ e' a oficial) ==="
move_if_exists "run_all.py"
move_if_exists "compare_basis.py"
move_if_exists "fried_nemenyi.py"
move_if_exists "audit_runs.py"
move_if_exists "merge_consolidated.py"
move_if_exists "diagnostico.py"

echo ""
echo "=== 2. Codigo morto (nao referenciado pela arquitetura em uso) ==="
move_if_exists "src"
move_if_exists "run_experiment.py"
move_if_exists "reporting/compare_models.py"

echo ""
echo "=== 3. Scratch / exploratorio (job ja feito) ==="
move_if_exists "aux_temp.py"
move_if_exists "teste.ipynb"
move_if_exists "analyze_bases.py"

echo ""
echo "=== 4. Limpando __pycache__ (sempre seguro de remover, e' so' cache) ==="
find . -type d -name "__pycache__" -not -path "./archive/*" -exec rm -rf {} + 2>/dev/null
echo "  __pycache__ removidos (serao recriados automaticamente ao rodar o codigo)"

echo ""
echo "=================================================================="
echo "Reorganizacao concluida. Revise ${ARCHIVE_DIR}/ antes de deletar de"
echo "vez (ou simplesmente deixe la' -- nao atrapalha nada ficar arquivado)."
echo ""
echo "Arquivos que FICARAM e sao a pipeline oficial de analise estatistica:"
echo "  analysis/statistical_comparison.py   (Friedman + post-hoc Wilcoxon/Holm)"
echo "  analysis/ranking_report.py           (ranking + CSV consolidado, por dataset)"
echo "  analysis/check_basis_convergence.py  (checagem de empates exatos)"
echo "=================================================================="