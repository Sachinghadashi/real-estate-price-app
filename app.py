from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import pandas as pd

app = Flask(__name__)
CORS(app)  # allow React (different port) to call this API

# Load model and metadata
model = joblib.load("model.pkl")
with open("columns.json", "r") as f:
    columns_info = json.load(f)

locations = columns_info["locations"]

# For chart endpoint, load full CSV once
data = pd.read_csv("real_estate_data.csv")


@app.route("/locations", methods=["GET"])
def get_locations():
    """
    Returns list of available locations for dropdown in React.
    """
    return jsonify({"locations": locations})


@app.route("/predict", methods=["POST"])
def predict_price():
    """
    Expects JSON:
    {
      "area_sqft": 1200,
      "location": "Suburb",
      "bedrooms": 3,
      "bathrooms": 2,
      "house_age_years": 5
    }
    """
    try:
        input_data = request.get_json()

        area_sqft = float(input_data.get("area_sqft", 0))
        location = input_data.get("location", "")
        bedrooms = int(input_data.get("bedrooms", 0))
        bathrooms = int(input_data.get("bathrooms", 0))
        house_age_years = int(input_data.get("house_age_years", 0))

        # Prepare data as DataFrame with same columns as training
        df = pd.DataFrame([{
            "area_sqft": area_sqft,
            "location": location,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "house_age_years": house_age_years
        }])

        predicted_price = model.predict(df)[0]

        return jsonify({
            "predicted_price": round(float(predicted_price), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/price-trend", methods=["GET"])
def price_trend():
    """
    Returns average price per location for chart.
    Output:
    {
      "labels": ["City Center", "Suburb", ...],
      "data": [avg1, avg2, ...]
    }
    """
    try:
        grouped = data.groupby("location")["price_in_inr"].mean().reset_index()

        labels = grouped["location"].tolist()
        values = [round(float(x), 2) for x in grouped["price_in_inr"].tolist()]

        return jsonify({
            "labels": labels,
            "data": values
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
