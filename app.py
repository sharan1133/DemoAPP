from flask import Flask, render_template, request , request, jsonify
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import pickle
import joblib
import pandas as pd
import numpy as np  


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

with open('/home/sharanbalakrishnan/Desktop/Learning/APP/APP/data/rf_model.bin', 'rb') as f:
    RF = pickle.load(f)

with open('/home/sharanbalakrishnan/Desktop/Learning/APP/APP/data/mlp_model.bin', 'rb') as g:
    MLP = pickle.load(g)


@app.route('/api/predict', methods=['POST'])
def predict_api():
    data = request.get_json()
    dayofyear = int(data['dayofyear'])
    hour = int(data['hour'])
    dayofweek = int(data['dayofweek'])
    quarter = int(data['quarter'])
    month = int(data['month'])
    model = data['model']

    new_data = {
        'dayofyear':dayofyear,
        'hour':hour,
        'dayofweek':dayofweek,
        'quarter':quarter,
        'month':month
    }

    df = pd.DataFrame([new_data])

    if model == 'Random Forest':
        predicted_val = RF.predict(df)
    else:
        predicted_val = MLP.predict(df)

    pred = float(predicted_val[0])
    response = {'prediction': pred}

    return jsonify(response)







@app.route('/predict', methods=['POST' , 'GET'])
def predict():

    if request.method == 'POST':
        print(request.form)
        dayofyear = int(request.form['dayofyear'])
        hour = int(request.form['hour'])
        dayofweek = int(request.form['dayofweek'])
        quarter = int(request.form['quarter'])
        month = int(request.form['month'])
        
        Model = request.form['MODEL']

        new_data = {
            'dayofyear':dayofyear,
            'hour':hour,
            'dayofweek':dayofweek,
            'quarter':quarter,
            'month':month
        }

        df = pd.DataFrame([new_data])



         

        if Model == 'Random Forest':
            predicted_val = RF.predict(df)
            pred = float(predicted_val[0])

        else:
            predicted_val = MLP.predict(df)
            pred = float(predicted_val[0])

        
        return render_template('index.html', prediction_text='Predicted Value is : {:.2f}'.format(pred))




       


