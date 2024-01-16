from flask import Flask ,request,render_template,jsonify
import numpy as np 
import pandas as pd
from flask  import Response
import pickle

application = Flask(__name__)
app = application

# importing model and scaler model into this app
model = pickle.load(open('Model\ModelForPrediction.pkl','rb'))
scaler = pickle.load(open('Model\StandardScaler.pkl','rb'))

#route for home page
@app.route('/')
def index():
    return render_template('index.html')


# route for single data point prediction
@app.route('/predictData' ,methods = ['GET','POST'])
def predict_datapoint():
    result = ""
    if request.method == 'POST':
        Pregnancies = float(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))
        


        new_scaled_data = scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict = model.predict(new_scaled_data)

        if predict[0] ==1:
            result = 'Diabetic'

        else:
            result = 'Non Diabetic'

                                     
        return render_template('result.html' ,result= result) 

    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.run(host = "0.0.0.0")