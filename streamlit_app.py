import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import holidays
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="ROAS Insights for Marketers",
    page_icon="💡",
    layout="wide"
)

# Title
st.title("💡 ROAS Insights Dashboard")
st.markdown(
    "Upload your campaign data to see expected returns, top-performing campaigns, "
    "and how changing budgets could boost your ROI."
)

# Sidebar: File upload
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader(
        "Choose your marketing CSV file",
        type=["csv"],
        help="Columns needed: date, campaign, channel, spend, installs, conversions, revenue, roas_day_1"
    )

if not uploaded_file:
    st.info("📂 Upload your campaign data CSV to get started.")
    st.stop()

# Load data
df = pd.read_csv(uploaded_file)

# Data preparation
def prepare(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "spend", "installs", "conversions", "revenue", "roas_day_1"])
    df["CAC"] = df["spend"] / df["installs"].replace(0, np.nan)
    df["CAC"] = df["CAC"].fillna(0)
    df["is_weekend"] = df["date"].dt.weekday >= 5
    us_holidays = holidays.US()
    df["is_holiday"] = df["date"].isin(us_holidays)
    df["campaign_code"] = df["campaign"].astype("category").cat.codes
    df["channel_code"] = df["channel"].astype("category").cat.codes
    df["spend_per_install"] = df["spend"] / (df["installs"] + 1)
    df["conversion_rate"] = df["conversions"] / (df["installs"] + 1)
    df["revenue_per_conversion"] = df["revenue"] / (df["conversions"] + 1)
    return df

df = prepare(df)

# Check required columns
required = ["date", "campaign", "channel", "spend", "installs", "conversions", "revenue", "roas_day_1"]
if any(col not in df.columns for col in required):
    missing = [col for col in required if col not in df.columns]
    st.error(f"Missing columns: {missing}")
    st.stop()

# Sidebar: Analysis options
with st.sidebar:
    st.header("2. Choose Analysis")
    option = st.selectbox(
        "Select what to view:",
        ["Overview", "ROAS Forecast", "Budget Simulator"]
    )

# 1. Overview
if option == "Overview":
    st.subheader("📊 Data Overview")
    st.metric("Days of Data", f"{len(df)}")
    st.metric("Unique Campaigns", df["campaign"].nunique())
    st.metric("Unique Channels", df["channel"].nunique())
    date_span = df["date"].max() - df["date"].min()
    st.metric("Data Time Span", f"{date_span.days} days")

    st.markdown("**Sample Data**")
    st.dataframe(df[required].head(10))

    st.markdown("**Key Averages**")
    avg_spend = df["spend"].mean()
    avg_conv = df["conversions"].mean()
    avg_roas1 = df["roas_day_1"].mean()
    st.write(f"- Average Daily Spend: ${avg_spend:,.0f}")
    st.write(f"- Average Daily Conversions: {avg_conv:,.0f}")
    st.write(f"- Average Day-1 ROAS: {avg_roas1:.2f}×")

# 2. ROAS Forecast (Day 1)
elif option == "ROAS Forecast":
    st.subheader("📈 Day-1 ROAS Prediction")
    features = [
        "spend", "installs", "conversions", "CAC",
        "is_weekend", "is_holiday",
        "campaign_code", "channel_code",
        "spend_per_install", "conversion_rate", "revenue_per_conversion"
    ]

    # Train model
    X = df[features]
    y = df["roas_day_1"]
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=50, random_state=42)
    model.fit(X, y)

    # Predict
    df["predicted_roas1"] = model.predict(X)

    # Show top campaigns
    st.markdown("**Top 5 Campaigns by Predicted Day-1 ROAS**")
    top = df.groupby("campaign")["predicted_roas1"].mean().nlargest(5)
    fig = px.bar(
        x=top.values, y=top.index,
        orientation="h",
        labels={"x": "Predicted ROAS (×)", "y": "Campaign"}
    )
    st.plotly_chart(fig, use_container_width=True)

# 3. Budget Simulator
else:
    st.subheader("💰 Budget Reallocation Simulator")
    st.markdown("Adjust spend percentages for each channel and see the impact on overall ROAS.")
    channels = df["channel"].unique()
    adjustments = {}
    cols = st.columns(2)
    for i, ch in enumerate(channels):
        adjustments[ch] = cols[i % 2].slider(
            f"{ch} spend change (%)", -50, 50, 0
        ) / 100

    if st.button("Run Simulation"):
        df_sim = df.copy()
        for ch, pct in adjustments.items():
            mask = df_sim["channel"] == ch
            df_sim.loc[mask, "spend"] *= (1 + pct)
        df_sim = prepare(df_sim)
        X_sim = df_sim[features]
        df_sim["predicted_roas1"] = model.predict(X_sim)

        original = (df["roas_day_1"] * df["spend"]).sum() / df["spend"].sum()
        new = (df_sim["predicted_roas1"] * df_sim["spend"]).sum() / df_sim["spend"].sum()
        uplift_pct = (new - original) / original * 100

        st.metric("Original Portfolio ROAS", f"{original:.2f}×")
        st.metric("New Portfolio ROAS", f"{new:.2f}×", delta=f"{new-original:.2f}×")
        st.write(f"**ROAS Improvement:** {uplift_pct:+.1f}%")

st.markdown("---")
st.caption("© Your Company Name")
