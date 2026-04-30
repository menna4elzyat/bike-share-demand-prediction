# 🚲 Bike Share Demand Prediction

A machine learning project that predicts hourly bike rental demand based on time, weather conditions, and previous demand values.

---

## 🌐 Live Demo

👉 Try the app here:  
https://bike-share-demand-prediction-q3fwgdogxlkt7frqdqswgm.streamlit.app/

---

## 📌 Project Overview

Bike-sharing systems generate large amounts of usage data.  
This project builds a regression model to predict bike demand using:

- Time features (hour, month, season)
- Weather conditions
- Historical demand (lag features)

The model is deployed using **Streamlit** for real-time predictions.

---

## 🧠 Key Features

- 📊 Full Exploratory Data Analysis (EDA)
- ⚙️ Feature Engineering (including lag features)
- 📉 Log Transformation for improved performance
- 🤖 Linear Regression Model
- 📈 Model Evaluation (RMSE, MAE, R²)
- 🌐 Interactive Web App using Streamlit

---

## 🔥 Model Insights

- Time-based features (especially hour) strongly influence demand
- Previous demand values (lag features) significantly improve predictions
- Weather conditions impact usage patterns
- The model achieves strong performance on both log and actual scales

---

## 🛠️ Tech Stack

- Python
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn
- Streamlit

---

## 📁 Project Structure
bike-share-demand-prediction/
│
├── app.py
├── reg.ipynb
├── bike_share_model.pkl
├── model_columns.pkl
├── requirements.txt
└── README.md

---

## ⚠️ Important Note

The model uses **lag features (t_1, t_2, t_3)**, which means:

> Previous demand values are required to make predictions.

In real-world scenarios, these values should be retrieved from historical data.

---

## 🚀 How to Run Locally

```bash


pip install -r requirements.txt

streamlit run app.py
