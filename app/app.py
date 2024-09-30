import joblib
from flask import Flask, request, jsonify
import numpy as np
from datetime import datetime

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('../data/model_27-09-2024-17-21-25-092478.pkl')

# Define a prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array([data['Store'], data['Promo'], data['StateHoliday'], data['SchoolHoliday'], 
                         data['DayOfWeek'], data['Month'], data['CompetitionDistance'], 
                         data['StoreType'], data['Assortment']]).reshape(1, -1)
    
    prediction = model.predict(features)[0]
    return jsonify({'predicted_sales': prediction})

if __name__ == '__main__':
    app.run(debug=True)