import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import requests
import joblib
from data_loader import load_data
from sequence_prep import SEQUENCE_LENGTH

# Load the test set and grab one engine's most recent 30 cycles
test = load_data('../archive/CMaps/test_FD001.txt')

feature_cols = [col for col in test.columns if col not in ['unit_number', 'time_cycles']]

# Let's use engine 1 from the test set
engine_1 = test[test['unit_number'] == 1].sort_values('time_cycles')

# Grab the LAST 30 cycles (most recent history) - this is what we'd have in real life
last_window = engine_1[feature_cols].tail(SEQUENCE_LENGTH).values

print("Window shape:", last_window.shape)

# Send it to the API
response = requests.post(
    'http://127.0.0.1:8000/predict',
    json={"readings": last_window.tolist()}
)

print("\nAPI Response:")
print(response.json())