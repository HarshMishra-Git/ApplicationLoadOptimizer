import streamlit as st
import pandas as pd
from app.data_processor import DataProcessor
from app.forecaster import Forecaster
from app.load_balancer import LoadBalancer
from app.metrics import MetricsCalculator
from app.utils import Visualizer

# Page config
st.set_page_config(
    page_title="Application Utilization Forecasting",
    page_icon="📊",
    layout="wide"
)

# Initialize components
@st.cache_resource
def init_components():
    data_processor = DataProcessor()
    forecaster = Forecaster()
    load_balancer = LoadBalancer()
    return data_processor, forecaster, load_balancer

data_processor, forecaster, load_balancer = init_components()

# Sidebar
st.sidebar.title("Configuration")
days_of_data = st.sidebar.slider("Days of Historical Data", 30, 180, 90)
forecast_hours = st.sidebar.slider("Forecast Hours", 24, 168, 48)

# Generate and process data
data = data_processor.generate_sample_data(days=days_of_data)
processed_data = data_processor.preprocess_data()
prophet_data = data_processor.prepare_prophet_data()

# Main content
st.title("Application Utilization Forecasting & Load Balancing")

# Forecasting section
st.header("Utilization Forecasting")
with st.spinner("Training forecasting model..."):
    forecaster.train(prophet_data)
    forecast = forecaster.predict(periods=forecast_hours)
    
    # Plot forecast
    forecast_plot = Visualizer.plot_forecast(forecast, prophet_data)
    st.plotly_chart(forecast_plot, use_container_width=True)
    
    # Metrics
    metrics = forecaster.get_metrics(
        prophet_data['y'][-forecast_hours:],
        forecast['yhat'][-forecast_hours:]
    )
    
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{metrics['MAE']:.2f}")
    col2.metric("MSE", f"{metrics['MSE']:.2f}")
    col3.metric("RMSE", f"{metrics['RMSE']:.2f}")

# Load Balancing section
st.header("Load Balancing Simulation")
if st.button("Simulate Load Balancing"):
    for _ in range(100):  # Simulate 100 requests
        load_balancer.round_robin(
            request_load=processed_data['server_load'].mean()
        )
    
    server_metrics = load_balancer.get_server_metrics()
    
    # Plot server loads
    load_plot = Visualizer.plot_server_loads(server_metrics)
    st.plotly_chart(load_plot, use_container_width=True)
    
    # Load distribution metrics
    distribution_metrics = MetricsCalculator.calculate_load_distribution(server_metrics)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Min Load", f"{distribution_metrics['min_load']:.2f}")
    col2.metric("Max Load", f"{distribution_metrics['max_load']:.2f}")
    col3.metric("Std Load", f"{distribution_metrics['std_load']:.2f}")
    col4.metric("Load Imbalance", f"{distribution_metrics['load_imbalance']:.2f}")

# Show raw data
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Application Usage Data")
    st.dataframe(processed_data)
