from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained model
try:
    model = pickle.load(open("tuned_rf_model.pkl", "rb"))
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Titanic Prediction API!", 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Validate required fields
        required_fields = ["Pclass", "Sex", "Age", "Fare"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400
        
        # Convert "Sex" (categorical) to numeric (male=1, female=0)
        sex_mapping = {"male": 1, "female": 0}
        if isinstance(data["Sex"], str):
            if data["Sex"].lower() not in sex_mapping:
                return jsonify({"error": "Invalid value for 'Sex'. Use 'male' or 'female'."}), 400
            data["Sex"] = sex_mapping[data["Sex"].lower()]

        # Convert input into NumPy array
        features = np.array([data["Pclass"], data["Sex"], data["Age"], data["Fare"]]).reshape(1, -1)

        # Ensure model is loaded
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        # Make prediction
        prediction = model.predict(features)

        # Return the result as JSON
        return jsonify({"Survived": int(prediction[0])}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Internal Server Error

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
