import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.fema_classifier import FEMaClassifier
from algebra.basis.basis import Basis
from algebra.neighboor_search.brute_force import BruteForceSearch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv('data/classificationData.csv', sep=';')

features = ['A', 'B', 'C']
target = ['class']

df = df[features + target].dropna()

train_x, test_x, train_y, test_y = train_test_split(
    df[features].values,
    df[target].values.ravel(),  # ravel() converte (n,1) para (n,) — necessário para np.unique no fit
    test_size=0.5
)

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# Shepard + BruteForce (euclidiana internamente)
basis = Basis.get('shepard', search=BruteForceSearch())
model = FEMaClassifier(basis=basis)

model.fit(train_x, train_y)

labels, probs = model.predict(test_x, k=10, z=2)

print(confusion_matrix(test_y, labels))