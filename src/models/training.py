import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor

# Load data and best params
X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").values.ravel()
params = joblib.load("models/best_params.pkl")

# Train model
model = GradientBoostingRegressor(**params)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/gbr_model.pkl")