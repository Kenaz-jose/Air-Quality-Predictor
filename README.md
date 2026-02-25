🌍 AI-Powered Air Quality Forecast System
An autonomous, deep-learning-based monitoring system that predicts PM2.5 concentrations using a Multivariate Gated Recurrent Unit (GRU) architecture.

📌 Project Overview
Traditional AQI monitors only provide real-time data. This project bridges the gap by providing predictive analytics. By analyzing 24-hour temporal patterns of wind, temperature, and humidity, the system forecasts air quality trends for the upcoming hour.

🧠 Technical Deep-Dive: Model Architecture
The core of this system is a GRU (Gated Recurrent Unit) network, specifically designed for multivariate time-series data.

Why GRU?
Temporal Memory: Maintains a hidden state to capture diurnal (24-hour) pollution cycles.

Efficiency: Features fewer parameters (Reset and Update gates) compared to LSTMs, enabling faster real-time inference on web-based dashboards.

Feature Engineering: Processes 16 refined features, including Sin/Cos time encoding and one-hot encoded calendar data to account for seasonal variations.

✨ Key Features
Zero-Touch Automation: Dual-integration with OpenWeatherMap APIs (Current Weather & Air Pollution) to fetch 23 parameters automatically.

Indian NAQI Standards: Automated categorization into Good, Satisfactory, Moderate, Poor, Very Poor, and Severe based on CPCB guidelines.

Geospatial Visualization: Integrated interactive map tracking the live location of the target city.

Residual Analytics: Real-time calculation of prediction error (Residuals) to compare AI forecasts against live sensor data.

💻 Tech Stack
Deep Learning: TensorFlow, Keras

Frontend: Streamlit

Data Processing: Pandas, NumPy, Scikit-Learn

Visualization: Plotly, Matplotlib

APIs: OpenWeatherMap