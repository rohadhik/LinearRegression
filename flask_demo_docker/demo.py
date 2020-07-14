import numpy as np
from flask import Flask, request, jsonify, Response
import pickle

app = Flask(__name__)
model = pickle.load(open('Pickle_RL_Model.pkl', 'rb'))

@app.route('/') 
def hello(): 
    return "welcome to the flask tutorials"
  

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    predict = model.predict([np.array(list(data.values()))])
    output = predict[0]
    return jsonify(output)

@app.route('/get_coefficients', methods=['GET'])
def get_coefficients():
    coefficient = str(model.coef_)
    return coefficient

@app.route('/get_intercept', methods=['GET'])
def get_intercept():
    intercept = str(model.intercept_)
    return intercept

if __name__ == "__main__": 
    app.run(host ='0.0.0.0', port = 5002, debug = True)
