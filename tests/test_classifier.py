import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.fema_classifier import FEMaClassifier
from algebra.basis.fem_basis import Basis
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'classificationData.csv'), sep=';')

features = ['A', 'B', 'C']
target = 'class'

df = df[features + [target]].dropna()

train_x, test_x, train_y, test_y = train_test_split(
    df[features].values,
    df[target].values,
    test_size=0.5
)

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

model = FEMaClassifier(k=2, z=10, basis=Basis.radialBasis)
model.fit(train_x, train_y)

pred, confidence_level = model.predict(test_x)

print(confusion_matrix(test_y, pred))