import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Charger les données
data = pd.read_csv("test.csv")

# Convertir les variables catégorielles en variables binaires avec get_dummies
data_encoded = pd.get_dummies(data)

# Calculer la matrice de corrélation
correlation_matrix = data_encoded.corr()

# Visualiser la matrice de corrélation avec un heatmap et enregistrer dans un fichier
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matrice de Corrélation')
plt.savefig('correlation_matrix.png')  # Enregistrer la figure dans un fichier