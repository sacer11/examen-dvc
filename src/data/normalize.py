import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
X_train = pd.read_csv("data/processed_data/X_train.csv")
X_test = pd.read_csv("data/processed_data/X_test.csv")

# Keep only numeric columns (drop e.g. datetime columns)
X_train = X_train.select_dtypes(include=["number"])
X_test = X_test.select_dtypes(include=["number"])

# Fit and transform
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save results
os.makedirs("data/processed_data", exist_ok=True)
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv("data/processed_data/X_train_scaled.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv("data/processed_data/X_test_scaled.csv", index=False)