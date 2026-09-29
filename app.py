from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from features import extract_url_features
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

print("Loading hybrid model trained on massive dataset...")
artifacts = joblib.load("phishing_rf_model_hybrid.joblib")
model = artifacts["model"]
feature_names = artifacts["feature_names"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    if not data or "url" not in data:
        return jsonify({"error": "Missing URL parameter"}), 400
        
    url = data["url"]
    
    try:
        # Extract the robust unbiased features from the raw URL
        feats = extract_url_features(url)
        X = pd.DataFrame([feats])
        X = X[feature_names] # Ensure column order matches training
        
        # Make Prediction
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1] # Probability of class 1 (Legitimate in PhiUSIIL)
        
        # Map labels (0: Phishing, 1: Legitimate)
        is_phishing = bool(pred == 0)
        phishing_probability = 1.0 - prob
        
        return jsonify({
            "url": url,
            "is_phishing": is_phishing,
            "phishing_probability": round(phishing_probability * 100, 2),
            "features": feats
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
