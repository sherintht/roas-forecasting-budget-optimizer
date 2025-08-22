import streamlit as st
import pandas as pd
import pickle
import os
from prophet import Prophet
import holidays

st.title('ROAS Forecasting Dashboard')

def load_model(target, models_dir='models'):
    with open(f"{models_dir}/{target}_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def get_features(df):
    us_holidays = holidays.US()
    df['date'] = pd.to_datetime(df['date'])
    df['CAC'] = df.apply(lambda x: x['spend'] / x['installs'] if x['installs'] > 0 else 0, axis=1)
    df['is_weekend'] = df['date'].dt.dayofweek >= 5
    df['is_holiday'] = df['date'].apply(lambda x: x in us_holidays)
    df['campaign_encoded'] = df['campaign'].astype('category').cat.codes
    df['channel_encoded'] = df['channel'].astype('category').cat.codes
    df['spend_per_install'] = df['spend'] / (df['installs'] + 1)
    df['conversion_rate'] = df['conversions'] / (df['installs'] + 1)
    df['revenue_per_conversion'] = df['revenue'] / (df['conversions'] + 1)
    return df

uploaded = st.file_uploader("Upload marketing_campaign_dataset.csv", type=['csv'])
if uploaded:
    df = pd.read_csv(uploaded)
    df = get_features(df)
    features = [
        'spend', 'installs', 'conversions', 'CAC', 'is_weekend', 'is_holiday',
        'campaign_encoded', 'channel_encoded', 'spend_per_install', 'conversion_rate', 'revenue_per_conversion'
    ]
    model = load_model('roas_day_1')
    st.subheader('Top Campaign Recommendations')
    preds = model.predict(df[features])
    df['predicted_roas'] = preds
    top_campaigns = df.sort_values('predicted_roas', ascending=False).groupby('campaign').first().reset_index()
    st.write(top_campaigns[['campaign','predicted_roas']].head(5))
    
    st.subheader('Prophet Forecast (Demo)')
    forecast_input = st.selectbox('Select campaign', df['campaign'].unique())
    ts_df = df[df['campaign']==forecast_input].copy()
    ts_df = ts_df.groupby('date')['revenue'].sum().reset_index().rename(columns={'date':'ds','revenue':'y'})
    m = Prophet()
    m.add_country_holidays('US')
    m.fit(ts_df)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    st.line_chart(forecast[['ds','yhat']].set_index('ds'))
    
    if st.checkbox('Run Budget Reallocation Simulator'):
        channel = st.selectbox('Select channel', df['channel'].unique())
        rel_shift = st.slider('Change (%)', -50, 50, 0)
        mask = df['channel'] == channel
        df.loc[mask, 'spend'] *= (1 + rel_shift/100.0)
        df['CAC'] = df.apply(lambda x: x['spend'] / x['installs'] if x['installs'] > 0 else 0, axis=1)
        sim_preds = model.predict(df[features])
        new_portfolio_roas = (sim_preds * df['spend']).sum() / df['spend'].sum()
        original_roas = (preds * df['spend']).sum() / df['spend'].sum()
        uplift = new_portfolio_roas - original_roas
        st.write(f"Original portfolio ROAS: {original_roas:.4f}")
        st.write(f"New portfolio ROAS: {new_portfolio_roas:.4f}")
        st.write(f"ROAS uplift: {uplift:.4f}")

st.info('For actual use, run the main pipeline first to train and save models, then use the dashboard on new data.')
