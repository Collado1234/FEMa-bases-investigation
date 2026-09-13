import json, glob

context = "classifier"
basis = "attention"
dataset = "blood_transfusion_service_center"
experiment = "experiments_v2"

scope = f"results/{context}/{basis}/{dataset}/{experiment}"

# 1. o que o summary.json diz que e' o combo vencedor
with open(f"{scope}/summary.json", encoding="utf-8") as f:
    summary = json.load(f)
winning_combo_id = summary["best_configuration"]["combo_id"]
print(f"combo_id vencedor (summary.json): {winning_combo_id!r}")

# 2. olhar todos os run_*.json e comparar combo_id + valor de f1
run_files = sorted(glob.glob(f"{scope}/run_*.json"))
print(f"\ntotal de run_*.json encontrados: {len(run_files)}")

combo_ids_found = set()
f1_values_for_winning = []

for rf in run_files:
    with open(rf, encoding="utf-8") as f:
        run = json.load(f)
    combo_ids_found.add(run.get("combo_id"))
    if run.get("combo_id") == winning_combo_id:
        f1_values_for_winning.append(run["metrics"].get("f1"))

print(f"\ncombo_ids distintos encontrados nos runs: {combo_ids_found}")
print(f"\nde {len(f1_values_for_winning)} runs do combo vencedor, valores de f1 (10 primeiros): "
      f"{f1_values_for_winning[:10]}")
print(f"quantos sao None: {sum(1 for v in f1_values_for_winning if v is None)} de {len(f1_values_for_winning)}")

# 3. mostrar um run completo do combo vencedor, cru
for rf in run_files:
    with open(rf, encoding="utf-8") as f:
        run = json.load(f)
    if run.get("combo_id") == winning_combo_id:
        print(f"\nexemplo de run completo ({rf}):")
        print(json.dumps(run, indent=2, ensure_ascii=False))
        break