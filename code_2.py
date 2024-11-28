import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# Étape 1 : Charger les données
file_path = r"./data/paris_fusion_sans_doublons.csv"
data = pd.read_csv(file_path)

# Étape 2 : Supprimer les lignes où le prix est NaN

data = data.dropna(subset=['price'])

# Étape 3 : Conversion du prix en numérique
data['price'] = data['price'].str.replace('$', '').str.replace(',', '').astype(float)

# Étape 4 : Gestion des valeurs manquantes
data['bathrooms'] = data['bathrooms'].fillna(1)
data['bedrooms'] = data['bedrooms'].fillna(1)
data['beds'] = data['beds'].fillna(1)

# Étape 5 : Conversion et recodage des colonnes
data['room_type'] = data['room_type'].map({
    'Entire home/apt': 0,
    'Entire rental unit': 0,
    'Entire serviced apartment': 0,
    'Private room': 1,
    'Shared room': 2,
    'Hotel room': 3,
    'Private room in bed and breakfast': 1,
    'Room in hotel': 3,
    'Private room in rental unit': 1
})

# Vérification après mapping
print("Valeurs uniques dans room_type :", data['room_type'].unique())

# Conversion des pourcentages
data['host_acceptance_rate'] = data['host_acceptance_rate'].str.rstrip('%').astype(float) / 100

# Sauvegarder les données nettoyées
data.to_csv(r"./data/data_nettoyee.csv", index=False)

# Étape 6 : Analyse descriptive (visualisation des prix)

plt.figure(figsize=(10, 6))
data['price'].hist(bins=200)
plt.title('Distribution des prix')
plt.xlabel('Prix')
plt.xlim(0, 1500) 
plt.ylabel('Nombre d\'annonces')
plt.savefig(r"./data/data_distribution_prix.png")
plt.close()

# Afficher les résultats
print("Données nettoyées enregistrées sous :", r"./data/data_nettoyee.csv")
print("Graphique de distribution des prix sauvegardé sous :", r"./data/data_distribution_prix.png")
