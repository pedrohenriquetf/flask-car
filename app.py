import numpy as np
from flask import Flask, jsonify, request, render_template

import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
vetor=['acc', 'good', 'unacc', 'vgood']

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    pred = model.predict(final_features)

    output = vetor[pred[0]]

    print(output)

    return render_template("index.html", prediction_text="CLASSIFICAÇÃO DO CARRO: " + output)


@app.route("/api", methods=["POST"])
def results():
    data = request.get_json(force=True)
    pred = model.predict([np.array(list(data.values()))])

    output = vetor[pred[0]]
    return jsonify(output)
