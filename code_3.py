# Import des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Étape 1 : Chargement des données
file_path = r"./data/data_nettoyee.csv"  # Ajustez le chemin si nécessaire
data = pd.read_csv(file_path)

# Étape 2 : Gestion des valeurs manquantes (rempli avec la moyenne dansn chaque colonnes respectives)
data['review_scores_cleanliness'] = data['review_scores_cleanliness'].fillna(data['review_scores_cleanliness'].mean())
data['review_scores_location'] = data['review_scores_location'].fillna(data['review_scores_location'].mean())
data['review_scores_rating'] = data['review_scores_rating'].fillna(data['review_scores_rating'].mean())

# Étape 3 : Suppression des valeurs aberrantes
data = data[(data['price'] > 10) & (data['price'] < 2000)]

# Étape 4 : Suppression des colonnes non pertinentes ou non numériques
colonnes_a_supprimer = ['id', 'description', 'last_review', 'amenities', 'neighborhood_overview', 'host_location']
data = data.drop(columns=colonnes_a_supprimer, errors='ignore')

# Étape 5 : Réduction des catégories (simplification pour certaines colonnes , enleve ceux qui ont moins de 50 occurences)
def reduce_categories(column, threshold=50):
    freq = column.value_counts()
    return column.apply(lambda x: x if freq[x] > threshold else "Other")

data['property_type'] = reduce_categories(data['property_type'])

# Étape 6 : Encodage des variables catégoriques ( transforme en valeur numérique)
colonnes_categoriques = ['property_type', 'neighbourhood_cleansed', 'room_type']  # Ajoutez d'autres colonnes si nécessaire
data_encoded = pd.get_dummies(data, columns=colonnes_categoriques, drop_first=True)

# Étape 7 : Remplissage des valeurs manquantes restantes
data_encoded = data_encoded.fillna(0)

# Étape 8 : Séparation des données en X (features) et y (target)
X = data_encoded.drop(columns=['price'])
y = data_encoded['price']

# Étape 9 : Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Étape 10 : Régression Linéaire Simple (basée sur 'accommodates')
X_train_simple = X_train[['accommodates']]
X_test_simple = X_test[['accommodates']]

model_lr_simple = LinearRegression()
model_lr_simple.fit(X_train_simple, y_train)
y_pred_lr_simple = model_lr_simple.predict(X_test_simple)
mse_lr_simple = mean_squared_error(y_test, y_pred_lr_simple)

# Étape 11 : Régression Linéaire Multiple
model_lr_multiple = LinearRegression()
model_lr_multiple.fit(X_train, y_train)
y_pred_lr_multiple = model_lr_multiple.predict(X_test)
mse_lr_multiple = mean_squared_error(y_test, y_pred_lr_multiple)

# Étape 12 : Modélisation avancée
# Modèle 1 : Forêt Aléatoire
model_rf = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)

# Modèle 2 : Gradient Boosting
model_gb = GradientBoostingRegressor(random_state=42, n_estimators=100, max_depth=5)
model_gb.fit(X_train, y_train)
y_pred_gb = model_gb.predict(X_test)
mse_gb = mean_squared_error(y_test, y_pred_gb)

# Étape 13 : Importance des variables
importances_rf = model_rf.feature_importances_
sorted_indices = np.argsort(importances_rf)[::-1]

# Affichage des importances
print("\nImportance des variables (Forêt Aléatoire) :")
for i in sorted_indices[:10]:
    print(f"{X.columns[i]}: {importances_rf[i]:.3f}")

# Visualisation 1 : Prix moyen par quartier
mean_price_by_neighbourhood = data.groupby('neighbourhood_cleansed')['price'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
mean_price_by_neighbourhood.head(10).plot(kind='bar', color='skyblue')
plt.title('Top 10 quartiers les plus chers')
plt.xlabel('Quartier')
plt.ylabel('Prix moyen')
plt.xticks(rotation=45)
plt.savefig(r"./data/mean_price_by_neighbourhood.png")
plt.close()

# Visualisation 2 : Prix en fonction du nombre d'accommodations
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['accommodates'], y=data['price'], alpha=0.5)
plt.title('Prix en fonction du nombre d\'accommodations')
plt.xlabel('Nombre d\'accommodations')
plt.ylabel('Prix')
plt.savefig(r"./data/price_vs_accommodates.png")
plt.close()

# Visualisation 3 : Matrice de corrélation
corr_matrix = data_encoded.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title("Matrice de corrélation")
plt.savefig(r"./data/correlation_matrix_final.png")
plt.close()

# Visualisation 4 : Comparaison des prédictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5, label='Prédictions Forêt Aléatoire')
plt.scatter(y_test, y_pred_gb, alpha=0.5, label='Prédictions Gradient Boosting', color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Valeurs réelles')
plt.xlabel("Valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.title("Comparaison des prédictions")
plt.legend()
plt.savefig(r"./data/comparison_predictions_final.png")
plt.close()

# Étape 14 : Résumé des résultats
print("\nMSE Régression Linéaire Simple : {:.2f}".format(mse_lr_simple))
print("MSE Régression Linéaire Multiple : {:.2f}".format(mse_lr_multiple))
print("MSE Forêt Aléatoire : {:.2f}".format(mse_rf))
print("MSE Gradient Boosting : {:.2f}".format(mse_gb))

print("\nRMSE Régression Linéaire Simple : {:.2f}".format(np.sqrt(mse_lr_simple)))
print("RMSE Régression Linéaire Multiple : {:.2f}".format(np.sqrt(mse_lr_multiple)))
print("RMSE Forêt Aléatoire : {:.2f}".format(np.sqrt(mse_rf)))
print("RMSE Gradient Boosting : {:.2f}".format(np.sqrt(mse_gb)))

print("\nR² Régression Linéaire Simple : {:.2f}".format(r2_score(y_test, y_pred_lr_simple)))
print("R² Régression Linéaire Multiple : {:.2f}".format(r2_score(y_test, y_pred_lr_multiple)))
print("R² Forêt Aléatoire : {:.2f}".format(r2_score(y_test, y_pred_rf)))
print("R² Gradient Boosting : {:.2f}".format(r2_score(y_test, y_pred_gb)))

print("\nLes résultats ont été sauvegardés dans './data/' :")
print("- Prix moyen par quartier : ./data/mean_price_by_neighbourhood.png")
print("- Prix en fonction du nombre d'accommodations : ./data/price_vs_accommodates.png")
print("- Matrice de corrélation : ./data/correlation_matrix_final.png")
print("- Comparaison des prédictions : ./data/comparison_predictions_final.png")
