import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI-Based NIDS", layout="wide")

st.title("🔐 AI-Based Network Intrusion Detection System")

st.write("""
This application uses Machine Learning to detect whether network traffic
is **Normal** or an **Intrusion**.
""")

# -------------------- DATA SIMULATION --------------------
def load_data():
    np.random.seed(42)
    data = {
        "packet_size": np.random.randint(100, 1500, 500),
        "duration": np.random.rand(500) * 10,
        "protocol": np.random.randint(0, 3, 500),
        "src_bytes": np.random.randint(50, 10000, 500),
        "dst_bytes": np.random.randint(50, 10000, 500),
        "label": np.random.randint(0, 2, 500)
    }
    return pd.DataFrame(data)

df = load_data()

X = df.drop("label", axis=1)
y = df["label"]

# -------------------- MODEL TRAINING --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)

if st.sidebar.button("Train Model Now"):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    st.sidebar.success(f"Model Trained Successfully ✅\nAccuracy: {acc:.2f}")

# -------------------- LIVE SIMULATOR --------------------
st.header("📡 Live Traffic Simulator")

packet_size = st.number_input("Packet Size", 100, 1500, 500)
duration = st.number_input("Duration", 0.0, 10.0, 1.0)
protocol = st.selectbox("Protocol (0-TCP, 1-UDP, 2-ICMP)", [0, 1, 2])
src_bytes = st.number_input("Source Bytes", 50, 10000, 200)
dst_bytes = st.number_input("Destination Bytes", 50, 10000, 300)

if st.button("Predict Traffic"):
    model.fit(X_train, y_train)
    sample = np.array([[packet_size, duration, protocol, src_bytes, dst_bytes]])
    result = model.predict(sample)[0]

    if result == 0:
        st.success("✅ Traffic is NORMAL")
    else:
        st.error("🚨 INTRUSION DETECTED!")

# -------------------- DATA VISUALIZATION --------------------
st.header("📊 Traffic Data Overview")

fig, ax = plt.subplots()
df["label"].value_counts().plot(kind="bar", ax=ax)
ax.set_xlabel("Traffic Type (0-Normal, 1-Attack)")
ax.set_ylabel("Count")
st.pyplot(fig)
