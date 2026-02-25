import streamlit as st
import numpy as np
import pandas as pd
import pickle
import requests
import math
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from dotenv import load_dotenv
import os

# 1. Configuration & Loading

# Load local .env file
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY") or st.secrets.get("OPENWEATHER_API_KEY")
WINDOW_SIZE = 24
NUM_FEATURES = 16

@st.cache_resource
def load_resources():
    model = load_model("model/gru_pm25_model.h5")
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_resources()

if "buffer" not in st.session_state:
    st.session_state.buffer = []

# 2. Logic Functions
def get_complete_data(city):
    w_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY.strip()}&units=metric"
    try:
        w_res = requests.get(w_url).json()
        lat, lon = w_res["coord"]["lat"], w_res["coord"]["lon"]
        p_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY.strip()}"
        p_res = requests.get(p_url).json()
        return {
            "city": city,
            "live_pm25": p_res['list'][0]['components']['pm2_5'],
            "temp": w_res["main"]["temp"],
            "pres": w_res["main"]["pressure"],
            "dewp": w_res["main"]["temp"] - ((100 - w_res["main"]["humidity"]) / 5),
            "iws": w_res["wind"]["speed"],
            "deg": w_res.get("wind", {}).get("deg", 0),
            "lat": lat, "lon": lon
        }
    except Exception as e:
        st.error(f"Error fetching data for '{city}'. Please check the city name.")
        return None

def get_naqi_status(pm25):
    if pm25 <= 30: return "Good 🟢", "#2ECC71", "Minimal health impact."
    elif pm25 <= 60: return "Satisfactory 🟡", "#F1C40F", "Minor discomfort for sensitive people."
    elif pm25 <= 90: return "Moderate 🟠", "#E67E22", "Discomfort for heart/lung patients."
    elif pm25 <= 120: return "Poor 🔴", "#E74C3C", "Impact on most people on prolonged exposure."
    elif pm25 <= 250: return "Very Poor 🟣", "#8E44AD", "Respiratory illness likely."
    else: return "Severe 🟤", "#5D4037", "Serious health impact on everyone."

def prepare_input_data(pm25, data):
    now = datetime.now()
    h, m, d_name = now.hour, now.month, now.strftime("%A")
    h_sin, h_cos = math.sin(2*math.pi*h/24), math.cos(2*math.pi*h/24)
    m_sin, m_cos = math.sin(2*math.pi*m/12), math.cos(2*math.pi*m/12)
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    day_encoded = {f"day_{d}": (1 if d_name == d else 0) for d in days}
    row = {
        'pm25_processed': np.log1p(pm25), 'DEWP': data['dewp'], 'TEMP': data['temp'],
        'Iws': data['iws'], 'PRES': data['pres'], 'hour_sin': h_sin, 'hour_cos': h_cos,
        'month_sin': m_sin, 'month_cos': m_cos, **day_encoded
    }
    cols = ['pm25_processed', 'DEWP', 'TEMP', 'Iws', 'PRES', 'hour_sin', 'hour_cos', 
            'month_sin', 'month_cos', 'day_Monday', 'day_Tuesday', 'day_Wednesday', 
            'day_Thursday', 'day_Friday', 'day_Saturday', 'day_Sunday']
    return scaler.transform(pd.DataFrame([row], columns=cols))[0]

def inverse_pm25(scaled_val):
    dummy = np.zeros((1, NUM_FEATURES))
    dummy[0, 0] = scaled_val
    return np.expm1(scaler.inverse_transform(dummy)[0, 0])

# 3. Wide UI Layout
st.set_page_config(page_title="AQI Forecast Pro", layout="wide")

st.markdown("<h1 style='text-align: center;'>🌍 Autonomous Air Quality Forecast System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Real-time GRU-based PM2.5 Prediction</p>", unsafe_allow_html=True)

# ROW 1: Search and Control (Aligned in one layer)
st.write("---")
# We use vertical_alignment to ensure buttons don't "drop" below the input field
in1, in2, in3 = st.columns([2.5, 1, 1], vertical_alignment="bottom")

with in1:
    city_name = st.text_input("Enter City Name (e.g., Kochi, Thiruvananthapuram, Delhi)", value="Kochi")

with in2:
    btn = st.button("🚀 Fetch & Forecast", use_container_width=True)

with in3:
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.buffer = []
        st.rerun()

# Logic Execution
if btn:
    data = get_complete_data(city_name)
    if data:
        # Instant Demo Auto-Fill
        if not st.session_state.buffer:
            base_row = prepare_input_data(data['live_pm25'], data)
            st.session_state.buffer = [base_row + np.random.normal(0, 0.005, 16) for _ in range(23)]
        
        st.session_state.buffer.append(prepare_input_data(data['live_pm25'], data))
        if len(st.session_state.buffer) > WINDOW_SIZE: st.session_state.buffer.pop(0)

        # ROW 2: Map and Metrics
        st.write("---")
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.subheader(f"📍 Location Analysis: {city_name}")
            st.map(pd.DataFrame({'lat': [data['lat']], 'lon': [data['lon']]}), zoom=10)

        with col_right:
            # Predictions
            input_arr = np.array(st.session_state.buffer).reshape(1, 24, 16)
            pred_scaled = model.predict(input_arr)[0][0]
            final_pm25 = inverse_pm25(pred_scaled)
            status, color, advice = get_naqi_status(final_pm25)
            residual = final_pm25 - data['live_pm25']

            st.markdown(f"### Predicted Status: <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
            
            # Gauge and Health Info
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=final_pm25, title={'text': "Forecasted PM2.5", 'font':{'size':16}},
                    gauge={'axis': {'range': [0, 300]}, 'bar': {'color': color},
                           'steps': [{'range': [0, 30], 'color': "#2ECC71"}, {'range': [30, 60], 'color': "#F1C40F"},
                                     {'range': [60, 90], 'color': "#E67E22"}, {'range': [90, 120], 'color': "#E74C3C"}]}))
                fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)
            
            with m_col2:
                st.metric("Live API Level", f"{data['live_pm25']:.2f} µg/m³")
                st.metric("Model Residual", f"{residual:.2f} µg/m³", delta="Difference", delta_color="off")
                st.info(f"**Advice:** {advice}")

        # ROW 3: Detailed Weather & Trends
        st.write("---")
        t_col1, t_col2 = st.columns([1, 2])
        
        with t_col1:
            st.subheader("🌦️ Meteorological Data")
            st.table(pd.DataFrame({
                "Metric": ["Temp", "Humidity (DewP)", "Wind Speed", "Wind Deg"],
                "Value": [f"{data['temp']}°C", f"{data['dewp']:.2f}°C", f"{data['iws']} m/s", f"{data['deg']}°"]
            }))

        with t_col2:
            st.subheader("📊 24-Hour Sequential Trend")
            chart_data = [inverse_pm25(r[0]) for r in st.session_state.buffer]
            st.line_chart(chart_data)

# Footer Explainability Expander
st.write("---")
with st.expander("🧠 Model Information & Architecture"):
    st.write("This forecasting system uses a Multivariate **Gated Recurrent Unit (GRU)** network.")
    st.write("- **Input:** 24-hour meteorological window.")
    st.write("- **Automation:** Weather and Pollution APIs remove the need for manual data entry.")
    if st.checkbox("Show Neural Network Summary"):
        summary = []
        model.summary(print_fn=lambda x: summary.append(x))
        st.text("\n".join(summary))