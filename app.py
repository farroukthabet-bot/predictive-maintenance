import streamlit as st
import requests
import sys
import os
import pandas as pd
import plotly.graph_objects as go

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data_loader import load_data
from sequence_prep import SEQUENCE_LENGTH

st.set_page_config(page_title="Engine RUL Predictor", page_icon="✈️")

st.title("✈️ Turbofan Engine Predictive Maintenance")
st.write("Select a test engine to see its predicted Remaining Useful Life (RUL).")

import os
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

# --- Load test data (cached so it's not reloaded on every interaction) ---
@st.cache_data
def get_test_data():
    test = load_data('archive/CMaps/test_FD001.txt')
    rul_answers = pd.read_csv('archive/CMaps/RUL_FD001.txt', sep=r'\s+', header=None, names=['RUL'])
    return test, rul_answers

test, rul_answers = get_test_data()
feature_cols = [col for col in test.columns if col not in ['unit_number', 'time_cycles']]

engine_ids = sorted(test['unit_number'].unique())
selected_engine = st.selectbox("Select Engine ID", engine_ids)

if st.button("Predict RUL"):
    engine_data = test[test['unit_number'] == selected_engine].sort_values('time_cycles')

    if len(engine_data) < SEQUENCE_LENGTH:
        st.error(f"Not enough history for this engine (needs {SEQUENCE_LENGTH} cycles).")
    else:
        last_window = engine_data[feature_cols].tail(SEQUENCE_LENGTH).values

        with st.spinner("Getting prediction from model..."):
            response = requests.post(API_URL, json={"readings": last_window.tolist()})

        if response.status_code == 200:
            result = response.json()
            predicted_rul = result['predicted_RUL']

            # The "true" answer, for comparison (we have this since it's the benchmark dataset)
            true_rul = rul_answers.iloc[selected_engine - 1]['RUL']

            col1, col2 = st.columns(2)
            col1.metric("Predicted RUL", f"{predicted_rul} cycles")
            col2.metric("Actual RUL", f"{true_rul} cycles", delta=f"{predicted_rul - true_rul:.1f} error")

            # Simple gauge-style visual for urgency
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predicted_rul,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Predicted Cycles Remaining"},
                gauge={
                    'axis': {'range': [0, 125]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 20], 'color': "red"},
                        {'range': [20, 50], 'color': "orange"},
                        {'range': [50, 125], 'color': "lightgreen"}
                    ]
                }
            ))
            st.plotly_chart(fig)

            # Show the sensor trend leading up to this point
            st.subheader("Sensor 11 trend (strongest predictor)")
            st.line_chart(engine_data.set_index('time_cycles')['sensor_11'])
        else:
            st.error(f"API error: {response.text}")