# application.py

from flask import Flask, request, render_template
import pickle
import numpy as np
import os

# Get the absolute path to the directory where this script (application.py) is located
# This will be C:\Users\ayush\OneDrive\Desktop\Algerian_fire_prediction
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Explicitly tell Flask where the templates folder is located.
# It will be BASE_DIR + /templates
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

# --- Diagnostic Prints (keep these for now, they are helpful!) ---
print(f"Flask app script path: {__file__}")
print(f"Calculated BASE_DIR: {BASE_DIR}")
print(f"Flask app's current working directory (os.getcwd()): {os.getcwd()}")
print(f"Flask's configured template folder: {app.template_folder}")
print(f"Attempting to load template from: {os.path.join(app.template_folder, 'index.html')}")
# --- End Diagnostic Prints ---


# Load your pickled scaler and model
# Use os.path.join with BASE_DIR for robust pathing to your models folder
try:
    with open(os.path.join(BASE_DIR, 'models', 'scaler.pkl'), 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    with open(os.path.join(BASE_DIR, 'models', 'ridgereg.pkl'), 'rb') as model_file:
        ridgereg = pickle.load(model_file)
    print("Models loaded successfully!")
except FileNotFoundError:
    print("ERROR: One or more model files not found. Please check paths and file existence.")
    print(f"Expected scaler path: {os.path.join(BASE_DIR, 'models', 'scaler.pkl')}")
    print(f"Expected ridgereg path: {os.path.join(BASE_DIR, 'models', 'ridgereg.pkl')}")
    scaler = None
    ridgereg = None
except Exception as e:
    print(f"An unexpected error occurred while loading models: {e}")
    scaler = None
    ridgereg = None


@app.route('/')
def home():
    # Flask will now definitely look for 'index.html' inside the 'templates' folder
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get input data from your form
            # Ensure these names match the 'name' attributes in your index.html input fields
            features = [
                float(request.form['FFMC']),
                float(request.form['DMC']),
                float(request.form['DC']),
                float(request.form['ISI']),
                float(request.form['temp']),
                float(request.form['RH']),
                float(request.form['BUI']),
                float(request.form['wind']),
                float(request.form['rain'])
            ]
            final_features = np.array(features).reshape(1, -1) # Reshape for a single prediction

            # Scale the input features
            if scaler:
                scaled_features = scaler.transform(final_features)
            else:
                return render_template('index.html', prediction_text="Error: Scaler not loaded.")

            # Make prediction
            if ridgereg:
                prediction = ridgereg.predict(scaled_features)[0] # Get the first (and only) prediction
                output = round(prediction, 2)
                return render_template('index.html', prediction_text=f"Predicted FWI: {output}")
            else:
                return render_template('index.html', prediction_text="Error: Model not loaded.")

        except Exception as e:
            # If there's an error processing input or prediction, display it
            return render_template('index.html', prediction_text=f"Error processing input: {e}")


if __name__ == '__main__':
    # Running in debug mode allows for automatic reloading on code changes
    # and provides a debugger in the browser if an error occurs.
    app.run(debug=True)