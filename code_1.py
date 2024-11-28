import pandas as pd

paris_juin = pd.read_csv(r"./data/listings_juin_brut.csv")
paris_septembre = pd.read_csv(r"./data/listings_septembre_brut.csv")
print("le fichier Paris_Juin contient", paris_juin.shape[0], "lignes et", paris_juin.shape[1], "colonnes.") 
print("le fichier Paris_Septembre contient", paris_septembre.shape[0], "lignes et", paris_septembre.shape[1], "colonnes.")

colonnes_a_garder = [
    'price', 'id', 'host_total_listings_count', 'property_type', 'room_type', 
    'accommodates', 'description', 'host_acceptance_rate', 'bathrooms', 'bedrooms', 'beds', 
    'number_of_reviews_ltm', 'last_review', 'review_scores_rating', 
    'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 
    'review_scores_communication', 'review_scores_location', 'review_scores_value', 'reviews_per_month', 'amenities', 'neighbourhood_cleansed'
]

paris_juin1 = paris_juin.filter(items=colonnes_a_garder)
print(paris_juin1)

paris_juin1.to_csv(r"./data/paris_juin1.csv", index=False)

paris_septembre1 = paris_septembre.filter(items=colonnes_a_garder)
print(paris_septembre1)

paris_septembre1.to_csv(r"./data/paris_septembre1.csv", index=False)


# Charger les fichiers CSV
juin = pd.read_csv(r"./data/paris_juin1.csv")
septembre = pd.read_csv(r"./data/paris_septembre1.csv")


# Combiner les deux fichiers
fusion = pd.concat([juin, septembre])

# Supprimer les doublons basés sur l'ID, en gardant la dernière occurrence (septembre)
fusion_sans_doublons = fusion.drop_duplicates(subset='id', keep='last')


# Sauvegarder le résultat dans un nouveau fichier CSV
fusion_sans_doublons.to_csv(r"./data/paris_fusion_sans_doublons.csv", index=False)

print("Fichier fusionné avec colonnes organisées enregistré sous 'paris_fusion_sans_doublons.csv'")
