import pandas as pd
import joblib
import json
from sklearn.metrics import mean_squared_error, r2_score

# Load model and data
model = joblib.load("models/gbr_model.pkl")
X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv").values.ravel()

# Predict
y_pred = model.predict(X_test)

# Save predictions
pd.DataFrame(y_pred, columns=["prediction"]).to_csv("data/processed_data/prediction.csv", index=False)

# Calculate metrics
metrics = {
    "mse": mean_squared_error(y_test, y_pred),
    "r2": r2_score(y_test, y_pred)
}

# Save metrics
with open("metrics/scores.json", "w") as f:
    json.dump(metrics, f, indent=4)