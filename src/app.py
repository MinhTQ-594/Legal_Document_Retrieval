from flask import Flask, request, jsonify
from flask_cors import CORS
from model import predict

import time

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def get_prediction():
    start = time.time()
    data = request.get_json()
    query = data.get('query')
    model_name = data.get('model')

    if not query or not model_name:
        return jsonify({'error': 'Query and model are required'}), 400

    try:
        result = predict(query, model_name)
        print(f"Time taken: {time.time() - start:.2f} seconds")
        return jsonify({'law_id': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return jsonify("Welcome to the Law Prediction API!")

if __name__ == "__main__":
    app.run(debug=True)
