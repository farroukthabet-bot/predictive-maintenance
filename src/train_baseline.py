import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

from data_loader import load_data
from feature_engineering import add_rul_column, cap_rul

# Load and prep the data
train = load_data('../archive/CMaps/train_FD001.txt')
train = add_rul_column(train)
train = cap_rul(train)

# Features = everything except identifiers and the target itself
feature_cols = [col for col in train.columns if col not in ['unit_number', 'time_cycles', 'RUL']]

X = train[feature_cols]
y = train['RUL']

# Split into train/validation sets (80/20) so we can honestly check performance
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_val)
mae = mean_absolute_error(y_val, preds)
rmse = np.sqrt(mean_squared_error(y_val, preds))

print(f"Baseline Random Forest Results:")
print(f"MAE (Mean Absolute Error): {mae:.2f} cycles")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f} cycles")

# Which sensors mattered most? Useful insight, and a good sanity check
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 most important features:")
print(importances.head(10))