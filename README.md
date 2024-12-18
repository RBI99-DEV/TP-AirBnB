# TP-AirBnB

## Structure du projet
- **`data/`** : Contient les fichiers de données.
- **`README.md`** : Documentation du projet.

---

## Description du projet
Nous travaillons sur deux fichiers CSV contenant les informations AirBnB pour Paris (mois de Septembre et Juin). 

### Objectif :
L'objectif principal est de **traiter les données** et de produire une **prédiction des prix** pour les annonces AirBnB.

### Utilisation de Git LFS :
- Les fichiers CSV bruts sont volumineux, donc nous utilisons **Git LFS** pour les gérer.
- **Pré-requis :**
  - Installer Git LFS (si ce n'est pas encore fait).
  - Exécuter la commande suivante après avoir cloné le dépôt pour récupérer les fichiers bruts :
    ```bash
    git lfs pull
    ```
  - Cette étape est nécessaire pour faire fonctionner le **script 1**.

---

## Organisation des scripts
### **`Code_1.py`**
- Sélection des colonnes nécessaires.
- Fusion des deux fichiers CSV (`Septembre` et `Juin`) en un seul fichier.
- Suppression des doublons.

### **`Code_2.py`**
- Traitement des données (nettoyage, pré-traitement).
- Génération d'un graphique représentant la distribution des prix des annonces.

### **`Code_3.py`**
- Finalisation du traitement des données manquantes.
- Tests sur différents modèles de prédiction :
  - Régression Linéaire Simple.
  - Régression Linéaire Multiple.
  - Forêt Aléatoire.
  - Gradient Boosting.

---

## Auteur
- **Rémi Bisson**
- **Ilyes Eddart**
