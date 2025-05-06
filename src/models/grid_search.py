import os
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

# Load data
X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").values.ravel()

# Define model and parameter grid
model = GradientBoostingRegressor()
param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}

# GridSearch
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

# Save best parameters
os.makedirs("models", exist_ok=True)
joblib.dump(grid_search.best_params_, "models/best_params.pkl")