from flask import Flask, request
import sys

sys.path.append('src/models')
from predict_model import Predict
    
app = Flask(__name__)
predictor = Predict()

@app.route('/', methods = ['GET'])
def hello_world():
    return "<h2>Cognitive Engine API</h2><br><p>API to predict intent of a phrase.</p>"

@app.route('/predict', methods = ['GET'])
def predict():
    text = request.args.get('text')
    output = predictor.predict(text)
    return output

if __name__ == "__main__":
    app.run(debug = True)