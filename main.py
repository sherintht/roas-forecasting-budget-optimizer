import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import xgboost as xgb
from prophet import Prophet
import holidays

# Path to your dataset
DATA_PATH = r'D:\Internship\roas\marketing_campaign_dataset.csv'

# Directory to save models
MODELS_DIR = 'models'
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Load dataset
df = pd.read_csv(DATA_PATH)

# =================== Data Preparation & Feature Engineering ===================

def prepare_data(df):
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Example feature: Customer Acquisition Cost (CAC)
    df['CAC'] = df.apply(lambda x: x['spend'] / x['installs'] if x['installs'] > 0 else 0, axis=1)

    # Seasonality features: Is weekend
    df['is_weekend'] = df['date'].dt.dayofweek >= 5

    # US holidays
    us_holidays = holidays.US()
    df['is_holiday'] = df['date'].isin(us_holidays)

    # Convert categorical to codes
    for col in ['campaign', 'channel']:
        df[col] = df[col].astype('category').cat.codes

    return df

df = prepare_data(df)

# =================== Model Training for ROAS Prediction ===================

targets = ['roas_day_1', 'roas_day_7', 'roas_day_30']
feature_cols = ['spend', 'installs', 'conversions', 'CAC', 'is_weekend', 'is_holiday', 'campaign', 'channel']

models = {}  # To store trained models

for target in targets:
    X = df[feature_cols]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost regression
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    with open(f'{MODELS_DIR}/{target}_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Evaluation
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f'Model for {target} trained.')
    print(f'{target} - MAPE: {mape:.2f}, RMSE: {rmse:.2f}')

    models[target] = model

# =================== Time Series Forecasting with Prophet ===================

def forecast_campaign(df, campaign_id, channel, periods=30):
    # Filter data for specific campaign & channel
    df_filtered = df[(df['campaign'] == campaign_id) & (df['channel'] == channel)]
    ts_data = df_filtered.groupby('date').agg({'revenue':'sum', 'roas_day_1':'mean'}).reset_index()
    ts_data.rename(columns={'date':'ds', 'revenue':'y'}, inplace=True)

    # Create Prophet model with holidays
    us_holidays = holidays.US()
    holiday_df = pd.DataFrame({
        'holiday': 'US_holiday',
        'ds': list(us_holidays)
    })

    m = Prophet(holidays=holiday_df, country_holidays=holidays.CountryHolidays('US'))
    m.add_country_holidays(country_name='US')
    m.fit(ts_data)

    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# Save the models and forecasts for later use
print("All models trained and saved in the 'models' directory.")
print("Run 'streamlit_app.py' for interactive dashboard.")
