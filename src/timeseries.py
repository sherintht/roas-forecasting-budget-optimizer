import pandas as pd
import numpy as np
from prophet import Prophet

def fit_prophet_series(df_campaign, revenue_col="revenue_day30"):
    # df_campaign has columns: date, revenue_col
    dfp = df_campaign.rename(columns={"date":"ds", revenue_col:"y"})[["ds","y"]].copy()
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.5,
        seasonality_mode="multiplicative"
    )
    m.add_country_holidays(country_name="US")
    m.fit(dfp)
    return m

def forecast_prophet(m, horizon_days=28):
    future = m.make_future_dataframe(periods=horizon_days, freq="D", include_history=False)
    fc = m.predict(future)
    # Return prediction interval for uncertainty-aware business rules
    return fc[["ds","yhat","yhat_lower","yhat_upper"]].rename(columns={"ds":"date", "yhat":"forecast","yhat_lower":"p05","yhat_upper":"p95"})