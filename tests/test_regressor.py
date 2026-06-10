import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fem_regression import FEMaRegressor
from fem_basis import Basis

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt


# -----------------------------
# LOAD DATA (ROBUSTO)
# -----------------------------
df = pd.read_csv('data/regressionData.csv', sep=None, engine='python')

# limpa espaços invisíveis em nomes de colunas
df.columns = df.columns.str.strip()

features = ['MSSubClass', 'LotFrontage', 'LotArea', 'PoolArea', 'MoSold', 'YrSold']
target = 'SalePrice'

# validação rápida (evita erro silencioso)
missing_cols = set(features + [target]) - set(df.columns)
if missing_cols:
    raise ValueError(f"Colunas não encontradas no CSV: {missing_cols}\nColunas disponíveis: {df.columns.tolist()}")

df = df[features + [target]].dropna()

# -----------------------------
# SPLIT
# -----------------------------
X = df[features].values
y = df[target].values.ravel()

train_x, test_x, train_y, test_y = train_test_split(
    X, y,
    test_size=0.1,
    random_state=42
)

# -----------------------------
# SCALING
# -----------------------------
scaler = StandardScaler()

train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# -----------------------------
# MODEL (CONSISTENTE)
# -----------------------------
model = FEMaRegressor(
    k=3,
    basis=Basis.radialBasis
)

model.fit(train_x, train_y)

# -----------------------------
# PREDICTION (SEM k AQUI)
# -----------------------------
pred = model.predict(test_x)

# -----------------------------
# PLOT
# -----------------------------
plt.figure()
plt.plot(pred, c='r', label='Prediction')
plt.plot(test_y, c='b', label='True')
plt.legend()
plt.title("FEMa Regressor - Prediction vs True")
plt.show()

# -----------------------------
# METRIC
# -----------------------------
print("MAPE:", mean_absolute_percentage_error(test_y, pred))