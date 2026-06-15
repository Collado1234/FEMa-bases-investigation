import numpy as np
from ..src import FEMaClassifier, EuclideanDistance, BruteForceSearch, SheppardBasis

# 1. Preparar os dados de exemplo
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([0, 1, 0, 1])

X_test = np.array([[1.5, 2.5], [3.5, 4.5]])

# 2. Instanciar os componentes
distance_metric = EuclideanDistance()
search_method = BruteForceSearch()
interpolation_basis = SheppardBasis(distance=distance_metric, search=search_method)

# 3. Instanciar o classificador com a base
classifier = FEMaClassifier(basis=interpolation_basis)

# 4. Treinar o modelo
classifier.fit(X_train, y_train)

# 5. Fazer previsões
# k e z são passados como kwargs para o predict da base
predicted_labels = classifier.predict(X_test, k=2, z=2)
predicted_probabilities = classifier.predict_proba(X_test, k=2, z=2)

print("Rótulos preditos:", predicted_labels)
print("Probabilidades preditas:\n", predicted_probabilities)