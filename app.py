import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ==========================================================
# Load LSTM Model
# ==========================================================

model = load_model("rapido_taxi_demand_lstm.h5")

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv("rapido_hourly_taxi_demand.csv")

df["Hour"] = pd.to_datetime(df["Hour"])

# ==========================================================
# Prepare Scaler
# ==========================================================

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(
    df[["Taxi_Demand"]]
)

# ==========================================================
# Forecast Function
# ==========================================================

def predict_demand(hours):

    hours = int(hours)

    sequence = scaled_data[-24:]

    sequence = sequence.reshape(1,24,1)

    predictions = []

    for i in range(hours):

        pred = model.predict(
            sequence,
            verbose=0
        )

        predictions.append(pred[0,0])

        sequence = np.append(
            sequence[:,1:,:],
            pred.reshape(1,1,1),
            axis=1
        )

    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1,1)
    )

    predictions = predictions.flatten().round().astype(int)

    future = pd.DataFrame({

        "Hour":

        [f"Hour {i+1}" for i in range(hours)],

        "Predicted Taxi Demand":

        predictions

    })

    return future

# ==========================================================
# Business Recommendation
# ==========================================================

def business_recommendation(avg):

    if avg >= 60:

        return """
🚖 High Demand Expected

• Deploy additional drivers

• Activate surge pricing if required

• Monitor driver availability continuously
"""

    elif avg >= 40:

        return """
🚖 Moderate Demand Expected

• Maintain normal driver allocation

• Keep standby drivers available

• Monitor demand during peak hours
"""

    else:

        return """
🚖 Low Demand Expected

• Reduce idle drivers

• Schedule driver breaks

• Optimize fleet utilization
"""

# ==========================================================
# CSS
# ==========================================================

css = """

.gradio-container{

    max-width:1200px !important;

    margin:auto;

}

footer{

    visibility:hidden;

}

h1{

    text-align:center;

}

.summary{

    background:#fff7ef;

    padding:20px;

    border-radius:12px;

    border-left:6px solid orange;

    margin-bottom:15px;

}

.cta{

    text-align:center;

    background:#eaf8ea;

    padding:15px;

    border-radius:10px;

    font-size:18px;

    font-weight:bold;

}

"""
