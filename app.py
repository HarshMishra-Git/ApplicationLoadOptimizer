
import streamlit as st
import pandas as pd
import numpy as np
from app.data_processor import DataProcessor
from app.forecaster import Forecaster
from app.load_balancer import LoadBalancer
from app.utils import Visualizer

# Page config
st.set_page_config(
    page_title="Application Load Analyzer",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'forecaster' not in st.session_state:
    st.session_state.forecaster = Forecaster()
if 'load_balancer' not in st.session_state:
    st.session_state.load_balancer = LoadBalancer()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = Visualizer()

# Sidebar
st.sidebar.title("Configuration")
page = st.sidebar.selectbox("Select Page", ["Forecasting", "Load Balancing"])

# Data generation parameters
with st.sidebar.expander("Data Configuration"):
    days = st.number_input("Number of days", min_value=30, max_value=365, value=90)
    if st.button("Generate New Data"):
        st.session_state.data = st.session_state.data_processor.generate_sample_data(days=days)
        st.session_state.prophet_data = st.session_state.data_processor.prepare_prophet_data()

# Initialize data if not exists
if 'data' not in st.session_state:
    st.session_state.data = st.session_state.data_processor.generate_sample_data()
    st.session_state.prophet_data = st.session_state.data_processor.prepare_prophet_data()

if page == "Forecasting":
    st.title("📈 Application Usage Forecasting")
    
    # Display raw data
    with st.expander("View Raw Data"):
        st.dataframe(st.session_state.data)
    
    # Forecasting parameters
    col1, col2 = st.columns(2)
    with col1:
        forecast_periods = st.slider("Forecast Periods (hours)", 24, 168, 24)
    with col2:
        target_col = st.selectbox("Target Variable", ["active_users", "server_load", "response_time"])
    
    # Train and forecast
    if st.button("Generate Forecast"):
        with st.spinner("Training model and generating forecast..."):
            prophet_data = st.session_state.data_processor.prepare_prophet_data(target_col)
            st.session_state.forecaster.train(prophet_data)
            forecast = st.session_state.forecaster.predict(periods=forecast_periods)
            
            # Plot forecast
            fig = st.session_state.visualizer.plot_forecast(forecast, prophet_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show metrics
            metrics = st.session_state.forecaster.get_metrics(
                prophet_data['y'][-forecast_periods:],
                forecast['yhat'][-forecast_periods:]
            )
            st.subheader("Forecast Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{metrics['MAE']:.2f}")
            col2.metric("MSE", f"{metrics['MSE']:.2f}")
            col3.metric("RMSE", f"{metrics['RMSE']:.2f}")

elif page == "Load Balancing":
    st.title("⚖️ Load Balancing Simulation")
    
    # Load balancing parameters
    col1, col2 = st.columns(2)
    with col1:
        num_servers = st.slider("Number of Servers", 2, 10, 3)
        st.session_state.load_balancer = LoadBalancer(num_servers=num_servers)
    with col2:
        request_load = st.slider("Average Request Load", 10, 100, 50)
    
    # Simulate load balancing
    if st.button("Simulate Load Balancing"):
        with st.spinner("Simulating load balancing..."):
            # Simulate requests
            for _ in range(100):
                st.session_state.load_balancer.round_robin(request_load)
            
            # Get and display metrics
            server_metrics = st.session_state.load_balancer.get_server_metrics()
            
            # Plot server loads
            fig = st.session_state.visualizer.plot_server_loads(server_metrics)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display detailed metrics
            st.subheader("Server Metrics")
            cols = st.columns(len(server_metrics))
            for i, (server, metrics) in enumerate(server_metrics.items()):
                cols[i].metric(
                    f"Server {i}",
                    f"Load: {metrics['avg_load']:.1f}",
                    f"Response: {metrics['avg_response_time']:.1f}ms"
                )

# IF YOU WANT TO ADD LSTM OR SARIMA MODEL INSTEAD OF PROPHET MODIFY app.py 
# by following code
'''

import streamlit as st
import pandas as pd
import numpy as np
from app.data_processor import DataProcessor
from app.forecaster import Forecaster
from app.load_balancer import LoadBalancer
from app.utils import Visualizer

# Page config
st.set_page_config(
    page_title="Application Load Analyzer",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'forecaster' not in st.session_state:
    st.session_state.forecaster = Forecaster()
if 'load_balancer' not in st.session_state:
    st.session_state.load_balancer = LoadBalancer()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = Visualizer()

# Sidebar
st.sidebar.title("Configuration")
page = st.sidebar.selectbox("Select Page", ["Forecasting", "Load Balancing"])

# Data generation parameters
with st.sidebar.expander("Data Configuration"):
    days = st.number_input("Number of days", min_value=30, max_value=365, value=90)
    if st.button("Generate New Data"):
        st.session_state.data = st.session_state.data_processor.generate_sample_data(days=days)
        st.session_state.prophet_data = st.session_state.data_processor.prepare_prophet_data()

# Initialize data if not exists
if 'data' not in st.session_state:
    st.session_state.data = st.session_state.data_processor.generate_sample_data()
    st.session_state.prophet_data = st.session_state.data_processor.prepare_prophet_data()

if page == "Forecasting":
    st.title("📈 Application Usage Forecasting")
    
    # Display raw data
    with st.expander("View Raw Data"):
        st.dataframe(st.session_state.data)
    
    # Forecasting parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        forecast_periods = st.slider("Forecast Periods (hours)", 24, 168, 24)
    with col2:
        target_col = st.selectbox("Target Variable", ["active_users", "server_load", "response_time"])
    with col3:
        model_type = st.selectbox("Model Type", ["sarima", "lstm"])
        
    st.session_state.forecaster = Forecaster(model_type=model_type)
    
    # Train and forecast
    if st.button("Generate Forecast"):
        with st.spinner("Training model and generating forecast..."):
            prophet_data = st.session_state.data_processor.prepare_prophet_data(target_col)
            st.session_state.forecaster.train(prophet_data)
            forecast = st.session_state.forecaster.predict(periods=forecast_periods)
            
            # Plot forecast
            fig = st.session_state.visualizer.plot_forecast(forecast, prophet_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show metrics
            metrics = st.session_state.forecaster.get_metrics(
                prophet_data['y'][-forecast_periods:],
                forecast['yhat'][-forecast_periods:]
            )
            st.subheader("Forecast Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{metrics['MAE']:.2f}")
            col2.metric("MSE", f"{metrics['MSE']:.2f}")
            col3.metric("RMSE", f"{metrics['RMSE']:.2f}")

elif page == "Load Balancing":
    st.title("⚖️ Load Balancing Simulation")
    
    # Load balancing parameters
    col1, col2 = st.columns(2)
    with col1:
        num_servers = st.slider("Number of Servers", 2, 10, 3)
        st.session_state.load_balancer = LoadBalancer(num_servers=num_servers)
    with col2:
        request_load = st.slider("Average Request Load", 10, 100, 50)
    
    # Simulate load balancing
    if st.button("Simulate Load Balancing"):
        with st.spinner("Simulating load balancing..."):
            # Simulate requests
            for _ in range(100):
                st.session_state.load_balancer.round_robin(request_load)
            
            # Get and display metrics
            server_metrics = st.session_state.load_balancer.get_server_metrics()
            
            # Plot server loads
            fig = st.session_state.visualizer.plot_server_loads(server_metrics)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display detailed metrics
            st.subheader("Server Metrics")
            cols = st.columns(len(server_metrics))
            for i, (server, metrics) in enumerate(server_metrics.items()):
                cols[i].metric(
                    f"Server {i}",
                    f"Load: {metrics['avg_load']:.1f}",
                    f"Response: {metrics['avg_response_time']:.1f}ms"
                )

'''
