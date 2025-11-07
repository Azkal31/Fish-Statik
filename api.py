from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load('tuned_ridge_model_retrain.joblib')

# Define the fishing gear options based on the actual training data
fishing_gear_options = [
    'Bagan Tancap',
    'Bubu',
    'Jaring Hela Dasar', 
    'Jaring Insang Hanyut',
    'Jaring Insang Tetap',
    'Jaring Payang',
    'Lain-lain',
    'Pancing'
]

def preprocess_user_input(year, alat_tangkap, effort):
    """
    Preprocesses user input to be compatible with the trained model.
    
    Args:
        year (int): The year of the input.
        alat_tangkap (str): The type of fishing gear.
        effort (float): The number of trips.
    
    Returns:
        pd.DataFrame: The processed input data ready for prediction.
    """
    # Create input with exact feature names from model training
    data = {'Effort (trip)': [effort]}
    
    # Add one-hot encoded columns for fishing gear
    for gear in fishing_gear_options:
        column_name = f'Alat tangkap_{gear}'
        data[column_name] = [1 if alat_tangkap == gear else 0]
    
    # Create DataFrame with exact order as model expects
    processed_input = pd.DataFrame(data)
    
    # Ensure column order matches model training
    expected_columns = ['Effort (trip)'] + [f'Alat tangkap_{gear}' for gear in fishing_gear_options]
    processed_input = processed_input[expected_columns]
    
    return processed_input

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        year = int(data.get('year', 2024))
        alat_tangkap = data.get('alat_tangkap')
        effort = float(data.get('effort'))
        
        # Validate inputs
        if not alat_tangkap or alat_tangkap not in fishing_gear_options:
            return jsonify({'error': 'Invalid fishing gear type'}), 400
        
        if effort <= 0:
            return jsonify({'error': 'Effort must be positive'}), 400
        
        # Preprocess input
        processed_input = preprocess_user_input(year, alat_tangkap, effort)
        
        # Make prediction
        predicted_cpue = model.predict(processed_input)[0]
        
        # Calculate total production
        predicted_production = predicted_cpue * effort
        
        # Return results
        return jsonify({
            'success': True,
            'predicted_cpue': round(predicted_cpue, 2),
            'predicted_production': round(predicted_production, 2),
            'inputs': {
                'year': year,
                'alat_tangkap': alat_tangkap,
                'effort': effort
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/fishing-gear', methods=['GET'])
def get_fishing_gear():
    """Get available fishing gear options"""
    return jsonify({
        'fishing_gear_options': fishing_gear_options
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
