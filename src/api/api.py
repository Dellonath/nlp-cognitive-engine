from flask import Flask, request
from flask_cors import CORS
import sys

sys.path.append('src/models')
from predict_model import Predict
    
app = Flask(__name__)
CORS(app)

predictor = Predict()

@app.route('/', methods = ['GET'])
def hello_world():
    return "<h2>Cognitive Engine API</h2><br><p>API to predict target of a phrase.</p>"

@app.route('/predict', methods = ['GET'])
def predict():
    text = request.args.get('text')
    output = predictor.predict(text)
    return output

if __name__ == "__main__":
    app.run(debug = True)