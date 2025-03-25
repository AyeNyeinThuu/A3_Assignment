from flask import Flask, render_template, request
import numpy as np
import mlflow.pyfunc
import pickle
import os

app = Flask(__name__)

# Load necessary components for preprocessing
with open('brand_ohe.pkl', 'rb') as f:
    brand_ohe = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('poly.pkl', 'rb') as f:
    poly = pickle.load(f)

# Load MLflow model
model_path = os.path.abspath("../mlruns/1/0b5b7e962e774839a330a2a6c664a607/artifacts/model")
model = mlflow.pyfunc.load_model(model_path)

def predict_car_price(brand, max_power, year, fuel):
    encoded_brand = list(brand_ohe.transform([[brand]]).toarray()[0])
    sample = np.array([[max_power, year, fuel] + encoded_brand])
    sample[:, 0:2] = scaler.transform(sample[:, 0:2])
    sample = np.insert(sample, 0, 1, axis=1)
    sample_poly = poly.transform(sample)
    predicted_class = int(model.predict(sample_poly)[0])

    k_range = {
        0: '29999.0 - 260000.0',
        1: '260000.0 - 450000.0',
        2: '450000.0 - 680000.0',
        3: '680000.0 - 1000000.0'
    }

    predicted_label = k_range[predicted_class]
    return predicted_label

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    brand = request.form['brand']
    max_power = float(request.form['max_power'])
    year = int(request.form['year'])
    fuel = int(request.form['fuel'])  # Assuming 0 = Petrol, 1 = Diesel, etc.
    prediction = predict_car_price(brand, max_power, year, fuel)
    fuel_display = "Petrol" if fuel == 0 else "Diesel"

    return render_template(
        'result.html',
        prediction=prediction,
        brand=brand,
        year=year,
        fuel=fuel_display,
        max_power=max_power
    )

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
