from flask import Flask, request, jsonify, render_template
import numpy as numpy
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

def return_prediction(model, scaler, sample_json):
    s_len = sample_json['sepal_length']
    s_wid = sample_json['sepal_width']
    p_len = sample_json['petal_length']
    p_wid = sample_json['petal_width']

    flower = [[s_len, s_wid, p_len, p_wid]]
    flower = scaler.transform(flower)

    classes = ['setosa', 'versicolor', 'viriginica']

    class_ind = model.predict_classes(flower)
    return classes[class_ind[0]]



flower_model = load_model('final_iris_model.h5')
flower_scaler = joblib.load('iris_scaler.pkl')

@app.route("/")
def index():
    return render_template('home.ejs')

@app.route("/run", methods=['POST'])
def flower_prediction():
    contents = request.json
    results = return_prediction(flower_model, flower_scaler, contents)
    return render_template('prediction.ejs')

if __name__=='__main__':
    app.run()