import numpy as np
import pickle

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor

# Charger les données
data = pd.read_excel("C:/Users/user/PycharmProjects/website/2020.xlsx",header=0)
df = data.dropna().copy()  # Ajout de .copy() pour éviter SettingWithCopyWarning



# Encodage de la variable catégorielle "country"
le = LabelEncoder()
df.loc[:, 'country'] = le.fit_transform(df.loc[:, 'country'])

# Séparation des variables explicatives et de la variable cible
X = df.drop(['Total_arrivals', 'Average_length_of_stay'], axis=1)
y = df['Total_arrivals']

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Réduction de dimension avec ACP
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Séparation des données en données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_pca, y,
test_size=0.3, random_state=42)

# Entraînement du modèle XGBoost avec validation croisée
param_grid = {'learning_rate': [0.1, 0.01], 'max_depth': [3, 5, 7],
'n_estimators': [50, 100, 150]}
model = GridSearchCV(XGBRegressor(), param_grid, cv=5)
model.fit(X_train, y_train)

# Sauvegarde du modèle avec pickle
pickle.dump(model,open("C:/Users/user/PycharmProjects/website/model.pkl", "wb"))

def preprocess_input(input_features):
    df = pd.DataFrame([input_features], columns=['country', 'year','Number_of_establishments', 'Number_of_rooms','Occupancy_rate_rooms'])

    # Encodage de la variable catégorielle "country"
    df.loc[:, 'country'] = le.transform(df.loc[:, 'country'])

    # Normalisation des données
    df_scaled = scaler.transform(df)

    # Réduction de dimension avec ACP
    df_pca = pca.transform(df_scaled)

    return df_pca