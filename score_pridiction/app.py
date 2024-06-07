from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Load the IPL dataset
data = pd.read_csv('ipl_dataset.csv')

# Data preprocessing
# Assume you have already cleaned and preprocessed the dataset

# Feature selection
X = data[['runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5']]
y = data['total']  # Update the column name to the correct target variable name

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor()
model.fit(X_train, y_train)

@app.route('/')
def index():
    batting_teams = data['bat_team'].unique().tolist()
    bowling_teams = data['bowl_team'].unique().tolist()
    return render_template('index.html', batting_teams=batting_teams, bowling_teams=bowling_teams)

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs
    batting_team = request.form['batting_team']
    bowling_team = request.form['bowling_team']
    over = float(request.form['over'])
    runs = float(request.form['runs'])
    wickets = float(request.form['wickets'])
    last_5_over_runs = float(request.form['last_5_over_runs'])
    last_5_over_wickets = float(request.form['last_5_over_wickets'])
    actual_score = float(request.form['actual_score'])

    # Predict score using the trained model
    prediction = model.predict([[runs, wickets, over, last_5_over_runs, last_5_over_wickets]])

    return render_template('result.html', prediction=prediction[0], actual_score=actual_score)

if __name__ == '__main__':
    app.run(debug=True)
