import string
from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Load the model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Get the user's input from the form
    area_type=int(request.form['area_type'])
    location=int(request.form['location'])
    total_sqft=int(request.form['total_sqft'])
    bath=int(request.form['bath'])
    balcony = int(request.form['balcony'])
    bhk = int(request.form['bhk'])

    df = pd.DataFrame([[area_type, location, total_sqft, bath, balcony, bhk]], columns=['area_type', 'location', 'total_sqft', 'bath', 'balcony', 'bhk'])

    # Use the model to make a prediction
    prediction = model.predict(df)[0]

    # Return the prediction to the user
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)