
import streamlit as st
import pandas as pd
import pickle
import os
from prophet import Prophet

# Load models
@st.cache_resource
def load_models():
    models = {}
    model_dir = 'models'
    for filename in os.listdir(model_dir):
        if filename.endswith('_model.pkl'):
            target = filename.replace('_model.pkl', '')
            with open(os.path.join(model_dir, filename), 'rb') as f:
                models[target] = pickle.load(f)
    return models

models = load_models()

st.title("ROAS Forecasting & Budget Optimization Dashboard")
uploaded_file = st.file_uploader("Upload Your Campaign CSV Dataset", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Sample Data", df.head())

    # Prepare data as in training
    def prepare_data(df):
        df['date'] = pd.to_datetime(df['date'])
        df['CAC'] = df.apply(lambda x: x['spend'] / x['installs'] if x['installs'] > 0 else 0, axis=1)
        df['is_weekend'] = df['date'].dt.dayofweek >= 5
        us_holidays = holidays.US()
        df['is_holiday'] = df['date'].isin(us_holidays)
        for col in ['campaign', 'channel']:
            df[col] = df[col].astype('category').cat.codes
        return df

    import holidays
    df = prepare_data(df)

    # Show forecast options
    campaign_options = df['campaign'].unique()
    channel_options = df['channel'].unique()

    selected_campaign = st.selectbox("Select Campaign for Forecast", campaign_options)
    selected_channel = st.selectbox("Select Channel for Forecast", channel_options)
    forecast_days = st.slider("Forecast Days", 7, 60, 30)

    if st.button("Generate Forecast"):
        # Create future dataframe
        forecast_df = df[(df['campaign']==selected_campaign) & (df['channel']==selected_channel)]
        ts_data = forecast_df.groupby('date').agg({'revenue':'sum', 'roas_day_1':'mean'}).reset_index()
        ts_data.rename(columns={'date':'ds','revenue':'y'}, inplace=True)
        m = Prophet(country_holidays=holidays.CountryHolidays('US'))
        m.fit(ts_data)
        future = m.make_future_dataframe(periods=forecast_days)
        forecast = m.predict(future)
        st.line_chart(forecast.set_index('ds')['yhat'])

    # Example: Simulate budget change
    st.subheader("Simulate Budget Adjustment")
    adjust_spend = {}
    for ch in df['channel'].unique():
        adjust_spend[ch] = st.number_input(f"Adjust spend for channel {ch}", value=0.0)
    if st.button("Run Reallocation Simulation"):
        st.write("Simulation not implemented: placeholder for future enhancement.")

st.write("Please upload dataset to proceed.")
