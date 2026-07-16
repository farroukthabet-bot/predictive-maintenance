from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import joblib
import sys
import os

# Let this file import from src/
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

app = FastAPI(title="Engine RUL Predictor")

SEQUENCE_LENGTH = 30
N_FEATURES = 24

# --- Same model architecture as training - PyTorch needs this to load the weights ---
class RULPredictor(nn.Module):
    def __init__(self, n_features, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output).squeeze()

# --- Load the trained model and scaler once, when the server starts ---
model = RULPredictor(n_features=N_FEATURES)
model.load_state_dict(torch.load('src/lstm_model.pth'))
model.eval()

scaler = joblib.load('src/scaler.pkl')

# --- Define what a valid request looks like ---
class EngineData(BaseModel):
    readings: list[list[float]]  # shape: (30 cycles, 24 features)

@app.get("/")
def root():
    return {"message": "Engine RUL Predictor API is running"}

@app.post("/predict")
def predict_rul(data: EngineData):
    readings = np.array(data.readings)

    if readings.shape != (SEQUENCE_LENGTH, N_FEATURES):
        return {"error": f"Expected shape ({SEQUENCE_LENGTH}, {N_FEATURES}), got {readings.shape}"}

    # Scale the input the same way training data was scaled
    scaled = scaler.transform(readings)

    # Convert to the shape PyTorch expects: (batch_size=1, sequence_length, features)
    input_tensor = torch.FloatTensor(scaled).unsqueeze(0)

    with torch.no_grad():
        prediction = model(input_tensor).item()

    return {
        "predicted_RUL": round(prediction, 1),
        "interpretation": "cycles remaining before predicted failure (capped at 125)"
    }