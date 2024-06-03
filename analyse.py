import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Chargement des données
data = pd.read_csv("train.csv")

# Exploration et analyse des données (optionnel)
# Vous pouvez effectuer ici une analyse exploratoire des données pour mieux comprendre leur structure

# Prétraitement des données
# Suppression des colonnes non pertinentes pour la prédiction
data.drop(["PassengerId", "Name", "Cabin", "Destination"], axis=1, inplace=True)

# Remplacement des valeurs manquantes dans la colonne "Age" par la médiane
data["Age"] = data["Age"].fillna(data["Age"].median())

# Conversion des variables catégorielles en variables indicatrices (one-hot encoding)
data = pd.get_dummies(data)

# Séparation des données en features (X) et target (y)
X = data.drop("Transported", axis=1)
y = data["Transported"]

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construction et entraînement du modèle de prédiction (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Rapport de classification
print(classification_report(y_test, y_pred))