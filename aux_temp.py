import pandas as pd
df = pd.read_csv("consolidated_runs.csv")
print(df.groupby(["dataset", "basis", "combo_id"]).size().sort_values(ascending=False))