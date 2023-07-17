import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import pickle

# Load the data
try:
    df = pd.read_excel(r"C:/Users/user/PycharmProjects/website/formu_modifié.xlsx")
except FileNotFoundError:
    print("Le fichier n'a pas été trouvé")
    exit(1)

# Filter the dataframe to only contain the columns of interest
columns_of_interest = ["Type tourism", "Season", "interests traveling"]

if not all(column in df.columns for column in columns_of_interest):
    print("Certaines colonnes d'intérêt ne sont pas dans le dataframe")
    exit(1)

df = df[columns_of_interest]

# Create an instance of the OneHotEncoder
encoder = OneHotEncoder()

# Fit the encoder on the data and transform it
X_encoded = encoder.fit_transform(df[["Type tourism", "Season"]])

# Save the encoder
pickle.dump(encoder, open('encoder.pkl', 'wb'))

# Split the data into features (X) and target variable (y)
X = X_encoded.toarray()
y = df["interests traveling"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
clf = LogisticRegression()

try:
    clf.fit(X_train, y_train)
except ValueError as e:
    print(f"Erreur lors de l'entraînement du modèle : {e}")
    exit(1)

# Evaluate the model
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the model
pickle.dump(clf, open('reg.pkl', 'wb'))