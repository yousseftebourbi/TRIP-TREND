import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, session, flash , jsonify
import mysql.connector
import os
import pickle
from model import preprocess_input

from model import le, scaler, pca

app = Flask(__name__)
app.secret_key = os.urandom(24)

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="users"
)
cursor = conn.cursor()
model = pickle.load(open("model.pkl", "rb"))

model2 = pickle.load(open('reg.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

# création de la table users
cursor.execute("""CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                Name VARCHAR(255) NOT NULL,
                email VARCHAR(255) NOT NULL,
                password VARCHAR(255) NOT NULL)""")

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return redirect('/home')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def about():
    if request.method == 'POST':
        return redirect('/')
    return render_template('register.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
    if 'id' in session:
        return render_template('home.html')
    else:
        return redirect('/')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Récupérer les valeurs du formulaire
        features = [x for x in request.form.values()]

        # Préparer les données pour le modèle
        prepared_features = preprocess_input(features)

        # Prédire les résultats
        prediction = model.predict(prepared_features)
        return render_template('prediction.html', prediction_text='Total Arrival predicted  :  {} Arrival'.format(int(prediction[0])))

    return render_template('prediction.html')

@app.route('/predict2', methods=['GET', 'POST'])
def predict2():
    prediction_text = None
    selected_type = None
    selected_season = None

    if request.method == 'POST':
        data1 = request.form['type_tourism']
        data2 = request.form['season']
        selected_type = data1
        selected_season = data2


        # Create a DataFrame with the user input values
        if data1 and data2:
            df = pd.DataFrame({'Type tourism': [data1], 'Season': [data2]})

            # Encode the categorical variables using the trained encoder
            encoded_data = encoder.transform(df)

            # Make a prediction on the input data
            pred = model2.predict(encoded_data)
            prediction_text = f"The predicted traveler's interest is: {pred[0]}"

    return render_template('predict2.html', prediction_text=prediction_text , selected_type=selected_type , selected_season=selected_season)


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')


@app.route('/login_validation', methods=['POST'])
def login_validation():
    email = request.form.get('email')
    password = request.form.get('password')

    cursor.execute("""SELECT * FROM users WHERE email = %s AND password = %s""", (email, password))
    users = cursor.fetchall()
    if len(users) > 0:
        session['id']=users[0][0]
        return redirect('/home')
    else:
        flash('please check your login details and try again')
        return redirect('/')

@app.route('/add_user', methods=['POST'])
def add_user():
    name=request.form.get('uname')
    email=request.form.get('uemail')
    password=request.form.get('upassword')

    cursor.execute("""INSERT INTO `users` (`id`,`name`,`email`,`password`) VALUES (NULL,'{}','{}','{}')""".format(name,email,password))
    conn.commit()
    return redirect('/')

@app.route('/logout')
def logout():
    session.pop('id')
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)