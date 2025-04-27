from flask import Flask, request, jsonify, render_template
import pickle
import requests
import io
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Google Drive direct download links
RIDGE_MODEL_URL = 'https://drive.google.com/uc?id=1PXaqyFZQ94CGXwBmu4lwxsANqWrVMzuk' 
SCALER_URL = 'https://drive.google.com/uc?id=1mFEmXiYlulJw_ZoGoriy63KoWQh3RDZB' 

# Function to download and load pickle files from Google Drive
def load_pickle_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise error if download fails
    return pickle.load(io.BytesIO(response.content))

# Load models
ridge_model = load_pickle_from_url(RIDGE_MODEL_URL)
standard_scaler = load_pickle_from_url(SCALER_URL)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes')) 
        Region = float(request.form.get('Region')) 

        new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html', result=result)

    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
