import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("bike_share_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.set_page_config(
    page_title="Bike Share Demand Prediction",
    page_icon="🚲",
    layout="centered"
)

st.title("🚲 Bike Share Demand Prediction")
st.write("Predict hourly bike rental demand using time, weather, and previous demand values.")

st.markdown("### Input Features")

season = st.selectbox(
    "Season",
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "Spring",
        2: "Summer",
        3: "Fall",
        4: "Winter"
    }[x]
)

mnth = st.slider("Month", 1, 12, 6)
hr = st.slider("Hour", 0, 23, 12)

holiday = st.selectbox(
    "Holiday",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

weathersit = st.selectbox(
    "Weather Situation",
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "Clear / Few clouds",
        2: "Mist / Cloudy",
        3: "Light snow / Light rain",
        4: "Heavy rain / Snow"
    }[x]
)

temp = st.slider("Temperature Normalized", 0.0, 1.0, 0.5)
hum = st.slider("Humidity Normalized", 0.0, 1.0, 0.5)

st.markdown("### Previous Demand Values")
t_1 = st.number_input("Demand in previous hour (t_1)", min_value=0, value=100)
t_2 = st.number_input("Demand two hours ago (t_2)", min_value=0, value=100)
t_3 = st.number_input("Demand three hours ago (t_3)", min_value=0, value=100)

def prepare_input():
    input_dict = {
        "weathersit": weathersit,
        "temp": temp,
        "hum": hum,
        "t_1": np.log1p(t_1),
        "t_2": np.log1p(t_2),
        "t_3": np.log1p(t_3),
    }

    input_df = pd.DataFrame([input_dict])

    # Create dummy variables exactly like training
    input_df[f"season_{season}"] = 1
    input_df[f"holiday_{holiday}"] = 1
    input_df[f"mnth_{mnth}"] = 1
    input_df[f"hr_{hr}"] = 1

    # Match training columns
    final_input = pd.DataFrame(0, index=[0], columns=model_columns)

    for col in input_df.columns:
        if col in final_input.columns:
            final_input[col] = input_df[col]

    return final_input

if st.button("Predict Demand"):
    final_input = prepare_input()

    prediction_log = model.predict(final_input)
    prediction_actual = np.expm1(prediction_log)

    predicted_value = np.ravel(prediction_actual)[0]
    predicted_count = max(0, int(round(predicted_value)))

    st.success(f"Predicted Bike Demand: {predicted_count} bikes")

    with st.expander("Show model input"):
        st.dataframe(final_input)
