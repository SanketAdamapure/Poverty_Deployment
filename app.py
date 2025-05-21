# -*- coding: utf-8 -*-
"""
Created on Mon May 19 21:22:37 2025

@author: sanket
"""

from flask import Flask, request, jsonify
from flasgger import Swagger
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
swagger = Swagger(app)

@app.route('/')
def index():
    return 'API is running!', 200

try:
    print("🔁 Loading model...")
    classifier = load_model("poverty_NN_model_V2.h5")
    print("✅ Model loaded.")
except Exception as e:
    print("❌ Model loading failed:", e)
    import sys
    sys.exit(1)

@app.route('/predict', methods=['GET'])
def predict():
    """
    Predict poverty status using input features
    ---
    parameters:
      - name: pop_chng
        in: query
        type: number
        required: true
      - name: n_empld
        in: query
        type: number
        required: true
      - name: tax_rate
        in: query
        type: number
        required: true
      - name: pt_phone
        in: query
        type: number
        required: true
      - name: pt_rural
        in: query
        type: number
        required: true
      - name: age
        in: query
        type: number
        required: true
    responses:
      200:
        description: Prediction result
    """
    try:
        pop_chng = float(request.args.get("pop_chng"))
        n_empld = float(request.args.get("n_empld"))
        tax_rate = float(request.args.get("tax_rate"))
        pt_phone = float(request.args.get("pt_phone"))
        pt_rural = float(request.args.get("pt_rural"))
        age = float(request.args.get("age"))

        prediction = classifier.predict([[pop_chng, n_empld, tax_rate, pt_phone, pt_rural, age]])
        print("✅ Prediction:", prediction)

        return "Hello, the answer is " + str(prediction[0][0])
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
@app.route('/predict_json', methods=['POST'])
def predict_json():
    """
    Predict poverty status using JSON input
    ---
    parameters:
      - name: x
        in: body
        required: true
        schema:
          type: array
          items:
            type: number
    responses:
      200:
        description: Prediction result
    """
    try:
        data = request.get_json()
        x_input = np.array(data['x'])  # ✅ Convert to NumPy array
        prediction = classifier.predict(x_input)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


#if __name__ == '__main__':
    #print("🚀 Starting Flask server...")
    #app.run(debug=True, port=5000, use_reloader=False)  # Fix added here
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # default to 10000 if PORT not set
    app.run(host="0.0.0.0", port=port)

