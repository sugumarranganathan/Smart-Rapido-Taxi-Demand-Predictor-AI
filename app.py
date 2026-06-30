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

# ==========================================================
# Generate Forecast
# ==========================================================

def generate_forecast(hours):

    forecast = predict_demand(hours)

    average = int(forecast["Predicted Taxi Demand"].mean())

    maximum = int(forecast["Predicted Taxi Demand"].max())

    minimum = int(forecast["Predicted Taxi Demand"].min())

    summary = f"""
    <div class="summary">

    <h3>📊 Prediction Summary</h3>

    <b>Average Demand :</b> {average}<br><br>

    <b>Highest Demand :</b> {maximum}<br><br>

    <b>Lowest Demand :</b> {minimum}

    </div>
    """

    recommendation = business_recommendation(average)

    recommendation = f"""

    <div class="summary">

    <h3>💼 Business Recommendation</h3>

    <pre>{recommendation}</pre>

    </div>

    """

    plt.figure(figsize=(10,4))

    plt.plot(

        forecast["Hour"],

        forecast["Predicted Taxi Demand"],

        marker="o",

        linewidth=2

    )

    plt.title("Future Taxi Demand Forecast")

    plt.xlabel("Forecast Hours")

    plt.ylabel("Predicted Taxi Demand")

    plt.grid(True)

    plt.tight_layout()

    image = "forecast.png"

    plt.savefig(image)

    plt.close()

    csv_file = "forecast.csv"

    forecast.to_csv(

        csv_file,

        index=False

    )

    return (

        summary,

        forecast,

        image,

        recommendation,

        csv_file

    )

# ==========================================================
# User Interface
# ==========================================================

with gr.Blocks(

    title="Smart Rapido Taxi Demand Predictor",

    css=css

) as demo:

    gr.Markdown("""

# 🚖 Smart Rapido Taxi Demand Predictor

Predict future taxi demand using an **LSTM Deep Learning Model**.

""")

    with gr.Row():

        hours = gr.Dropdown(

            choices=[6,12,24,48],

            value=24,

            label="📅 Forecast Hours"

        )

        predict_btn = gr.Button(

            "🚀 Predict Taxi Demand",

            variant="primary"

        )

        clear_btn = gr.ClearButton()

    summary = gr.HTML()

    with gr.Row():

        table = gr.Dataframe(

            label="📋 Forecast Table"

        )

        graph = gr.Image(

            label="📈 Demand Forecast"

        )

    recommendation = gr.HTML()

    download = gr.File(

        label="📥 Download Forecast CSV"

    )

    predict_btn.click(

        fn=generate_forecast,

        inputs=hours,

        outputs=[

            summary,

            table,

            graph,

            recommendation,

            download

        ]

    )

demo.queue()

demo.launch()


