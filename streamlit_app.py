import streamlit as st
import pandas as pd
from pathlib import Path

def run_streamlit_app():
    st.title("ROAS Forecasting and Optimization Dashboard")
    uploaded_file = st.file_uploader("Upload your marketing campaign dataset", type=['csv'])
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        data = prepare_data(data)
        
        st.subheader("Raw Data")
        st.write(data.head())
        
        st.subheader("ROAS Prediction Metrics")
        for target in targets:
            st.write(f"{target} MAPE: {results[target]['MAPE']:.4f}")
            st.write(f"{target} RMSE: {results[target]['RMSE']:.4f}")
        
        # Select campaign and channel for forecasting
        campaign_sel = st.selectbox("Select Campaign for Forecast", data['campaign'].unique())
        channel_sel = st.selectbox("Select Channel for Forecast", data['channel'].unique())
        periods = st.slider("Select Forecast Horizon (days)", 7, 60, 30)
        
        if st.button("Generate Forecast"):
            forecast = prophet_forecast(data, campaign_sel, channel_sel, periods=periods)
            st.line_chart(forecast.set_index('ds')['yhat'])
        
        # Budget reallocation simulation input
        st.subheader("Budget Reallocation Simulator")
        channels = list(data['channel'].unique())
        spend_shifts = {}
        for ch in channels:
            spend_shifts[ch] = st.number_input(f"Budget shift for channel {ch}", value=0.0)
        
        if st.button("Simulate Budget Reallocation"):
            before, after, uplift = budget_reallocation_simulator(data, results, spend_shifts)
            st.write(f"Current Portfolio ROAS: {before:.4f}")
            st.write(f"Projected Portfolio ROAS after reallocation: {after:.4f}")
            st.write(f"Projected ROAS Uplift: {uplift:.4f}")

if __name__ == "__main__":
    run_streamlit_app()
