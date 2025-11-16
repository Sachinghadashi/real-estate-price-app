import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import json

# 1. Load dataset
data = pd.read_csv("real_estate_data.csv")

# 2. Features and target
X = data[["area_sqft", "location", "bedrooms", "bathrooms", "house_age_years"]]
y = data["price_in_inr"]

# 3. Categorical & numeric columns
categorical_features = ["location"]
numeric_features = ["area_sqft", "bedrooms", "bathrooms", "house_age_years"]

# 4. Preprocessor: OneHotEncode location, pass through numeric as-is
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)

# 5. Model: Linear Regression pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ]
)

# 6. Train-test split (optional, just to be proper)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Fit model
model.fit(X_train, y_train)

# 8. Save model
joblib.dump(model, "model.pkl")

# 9. Save metadata (columns & locations)
columns_info = {
    "categorical_features": categorical_features,
    "numeric_features": numeric_features,
    "locations": sorted(data["location"].unique().tolist())
}

with open("columns.json", "w") as f:
    json.dump(columns_info, f, indent=4)

print("Model trained and saved as model.pkl")
print("Columns metadata saved as columns.json")
