import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_loader import load_data
from feature_engineering import add_rul_column, cap_rul
from sequence_prep import create_sequences, SEQUENCE_LENGTH

# --- Load and prep data (same as before) ---
train = load_data('../archive/CMaps/train_FD001.txt')
train = add_rul_column(train)
train = cap_rul(train)

feature_cols = [col for col in train.columns if col not in ['unit_number', 'time_cycles', 'RUL']]

# --- Scale features (NEW - neural nets train much better on normalized data) ---
scaler = StandardScaler()
train[feature_cols] = scaler.fit_transform(train[feature_cols])

X, y = create_sequences(train, feature_cols)

# --- Train/validation split ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Convert to PyTorch tensors ---
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.FloatTensor(y_val)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# --- Define the LSTM model ---
class RULPredictor(nn.Module):
    def __init__(self, n_features, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        # We only care about the LAST time step's output for our prediction
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output).squeeze()

model = RULPredictor(n_features=X.shape[2])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- Training loop ---
EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.2f}")

# --- Evaluate on validation set ---
model.eval()
with torch.no_grad():
    val_preds = model(X_val_t).numpy()

mae = mean_absolute_error(y_val, val_preds)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))

import matplotlib.pyplot as plt

print(f"\nLSTM Results:")
print(f"MAE: {mae:.2f} cycles")
print(f"RMSE: {rmse:.2f} cycles")
print(f"Compare to baseline: MAE=13.57, RMSE=18.77")

# --- Save the model and scaler for later use (API serving) ---
torch.save(model.state_dict(), 'lstm_model.pth')
import joblib
joblib.dump(scaler, 'scaler.pkl')
print("\nModel and scaler saved.")

# --- Error analysis: does error get worse or better as true RUL decreases? ---
errors = np.abs(y_val - val_preds)

# Bucket predictions by true RUL range, see average error per bucket
buckets = [(0, 20), (20, 50), (50, 80), (80, 125)]
print("\nError by RUL range (this tells us where the model struggles):")
for low, high in buckets:
    mask = (y_val >= low) & (y_val < high)
    if mask.sum() > 0:
        bucket_mae = errors[mask].mean()
        print(f"  RUL {low}-{high}: MAE={bucket_mae:.2f} cycles (n={mask.sum()})")

# --- Plot predicted vs actual ---
plt.figure(figsize=(8, 6))
plt.scatter(y_val, val_preds, alpha=0.3)
plt.plot([0, 125], [0, 125], 'r--', label='Perfect prediction')
plt.xlabel('True RUL')
plt.ylabel('Predicted RUL')
plt.title('LSTM: Predicted vs True RUL')
plt.legend()
plt.savefig('lstm_predictions.png')
print("\nPlot saved as lstm_predictions.png")