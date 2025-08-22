import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import holidays
import plotly.express as px

# Page config
st.set_page_config(page_title="ROAS Dashboard (No Prophet)", page_icon="📊", layout="wide")

st.title("🚀 ROAS Forecasting & Optimization Dashboard")

# Sidebar for file upload
with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload campaign CSV", type=["csv"])

if not uploaded_file:
    st.info("Please upload a CSV with columns: date, campaign, channel, spend, installs, conversions, revenue, roas_day_1")
    st.stop()

# Load data
df = pd.read_csv(uploaded_file)

# Data preparation
def prepare(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["CAC"] = df.apply(lambda x: x["spend"] / x["installs"] if x["installs"] > 0 else 0, axis=1)
    df["is_weekend"] = df["date"].dt.dayofweek >= 5
    us_hols = holidays.US()
    df["is_holiday"] = df["date"].isin(us_hols)
    df["campaign_encoded"] = df["campaign"].astype("category").cat.codes
    df["channel_encoded"] = df["channel"].astype("category").cat.codes
    df["spend_per_install"] = df["spend"] / (df["installs"] + 1)
    df["conversion_rate"] = df["conversions"] / (df["installs"] + 1)
    df["revenue_per_conversion"] = df["revenue"] / (df["conversions"] + 1)
    return df

df = prepare(df)

# Feature list
features = [
    "spend", "installs", "conversions", "CAC",
    "is_weekend", "is_holiday",
    "campaign_encoded", "channel_encoded",
    "spend_per_install", "conversion_rate", "revenue_per_conversion"
]

# Validate columns
required = features + ["roas_day_1"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# Train XGBoost model
model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=50, random_state=42)
model.fit(df[features], df["roas_day_1"])
df["predicted_roas"] = model.predict(df[features])

# Top campaigns by predicted ROAS
st.subheader("Top 5 Campaigns by Predicted ROAS (Day 1)")
top = df.groupby("campaign")["predicted_roas"].mean().nlargest(5)
fig = px.bar(x=top.values, y=top.index, orientation="h",
             labels={"x": "Predicted ROAS", "y": "Campaign"})
st.plotly_chart(fig, use_container_width=True)

# Budget Reallocation Simulator
st.subheader("💰 Budget Reallocation Simulator")
channels = list(df["channel"].unique())
adjustments = {}
cols = st.columns(2)
for i, ch in enumerate(channels):
    adjustments[ch] = cols[i % 2].slider(f"{ch} change (%)", -50, 50, 0) / 100.0

if st.button("Run Simulation"):
    df_sim = df.copy()
    for ch, pct in adjustments.items():
        mask = df_sim["channel"] == ch
        df_sim.loc[mask, "spend"] *= (1 + pct)
    df_sim = prepare(df_sim)
    df_sim["predicted_roas"] = model.predict(df_sim[features])

    orig = (df["roas_day_1"] * df["spend"]).sum() / df["spend"].sum()
    new = (df_sim["predicted_roas"] * df_sim["spend"]).sum() / df_sim["spend"].sum()
    st.metric("Original Portfolio ROAS", f"{orig:.4f}")
    st.metric("New Portfolio ROAS", f"{new:.4f}", delta=f"{new - orig:.4f}")

st.markdown("---")
st.write("⚠️ Forecasting features removed to avoid Prophet installation issues. This app focuses on Day 1 ROAS prediction and budget simulation.")
