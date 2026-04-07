import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="Air Quality & Health Risk Monitor", page_icon="🌬️", layout="wide")

if "air_history" not in st.session_state:
    st.session_state.air_history = []

st.title("🌬️ Air Quality & Health Risk Monitor")
st.markdown("Classify air quality as **Safe (0) / Moderate (1) / Hazardous (2)** and predict respiratory risk.")

# ─── DATASET (FIXED) ───
np.random.seed(42)
n = 50

data = {
    "StationID": [f"ST{str(i).zfill(3)}" for i in range(1, n+1)],
    "City": (["Delhi","Mumbai","Bangalore","Hyderabad","Chennai"] * (n//5)),
    "Season": (["Winter","Summer","Monsoon","Autumn"] * (n//4 + 1))[:n],

    "PM25": np.random.choice([10, 20, 35, 80, 150, 250, None], n),
    "PM10": np.random.choice([20, 50, 100, 200, 400, None], n),
    "CO2ppm": np.random.choice([380, 400, 420, 450, 500, None], n),
    "Humidity": np.random.choice([40, 50, 60, 70, 80, None], n),
    "WindSpeedKmh": np.random.choice([2, 5, 10, 15, 20, None], n),
    "TemperatureC": np.random.choice([18, 22, 25, 30, 35, None], n),
}

df_raw = pd.DataFrame(data)

# Better target logic
df_raw["AQICategory"] = np.where(
    df_raw["PM25"].fillna(0) > 150, 2,
    np.where(df_raw["PM25"].fillna(0) > 35, 1, 0)
)

st.subheader("📋 Air Quality Dataset")
st.dataframe(df_raw)

col1, col2, col3 = st.columns(3)
col1.metric("Total Stations", len(df_raw))
col2.metric("Hazardous Days", (df_raw["AQICategory"] == 2).sum())
col3.metric("Safe Days", (df_raw["AQICategory"] == 0).sum())

# ─── PREPROCESSING ───
st.subheader("🔧 Data Preprocessing")

df = df_raw.copy()

# Fill missing
for col in ["PM25","PM10","CO2ppm","Humidity","WindSpeedKmh","TemperatureC"]:
    df[col] = df[col].fillna(df[col].median())

# Drop duplicates + ID
df = df.drop_duplicates().drop(["StationID"], axis=1)

# Encoding
df["City"] = LabelEncoder().fit_transform(df["City"])
df["Season"] = LabelEncoder().fit_transform(df["Season"])

# ─── FEATURE ENGINEERING ───
df["ParticulateMatter"] = df["PM25"] + df["PM10"]
df["PollutionIndex"] = df["PM25"] * df["CO2ppm"] / 10000
df["VentilationFactor"] = df["WindSpeedKmh"] / (df["Humidity"] + 1)
df["HeatHumidityIndex"] = df["TemperatureC"] * df["Humidity"] / 100

# ─── MODEL ───
st.subheader("🤖 SVM Multi-Class Classifier")

X = df.drop("AQICategory", axis=1)
y = df["AQICategory"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = SVC(kernel="rbf", probability=True, random_state=42)
model.fit(X_train, y_train)

st.success("✅ Model Trained Successfully!")

# ─── EVALUATION ───
st.subheader("📈 Evaluation")

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

col1, col2 = st.columns(2)
col1.metric("Accuracy", f"{acc*100:.2f}%")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
ax.imshow(cm)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i,j], ha="center", va="center")
ax.set_title("Confusion Matrix")
st.pyplot(fig)

st.text(classification_report(y_test, y_pred))

# ─── PREDICTION ───
st.subheader("🎯 Check Air Quality Risk")

city = st.selectbox("City", [0,1,2,3,4])
season = st.selectbox("Season", [0,1,2,3])
pm25 = st.slider("PM2.5", 0.0, 500.0, 35.0)
pm10 = st.slider("PM10", 0.0, 800.0, 60.0)
co2 = st.slider("CO2", 350, 600, 410)
humidity = st.slider("Humidity", 10, 100, 60)
wind = st.slider("Wind Speed", 0.0, 50.0, 10.0)
temp = st.slider("Temperature", -5, 50, 28)

if st.button("Predict"):
    part_mat = pm25 + pm10
    poll_idx = pm25 * co2 / 10000
    vent_fac = wind / (humidity + 1)
    heat_hum = temp * humidity / 100

    input_data = pd.DataFrame([[city, season, pm25, pm10, co2, humidity, wind, temp,
                                part_mat, poll_idx, vent_fac, heat_hum]],
                              columns=X.columns)

    input_scaled = scaler.transform(input_data)

    pred = model.predict(input_scaled)[0]

    if pred == 0:
        st.success("✅ SAFE")
    elif pred == 1:
        st.warning("⚠️ MODERATE")
    else:
        st.error("🚨 HAZARDOUS")