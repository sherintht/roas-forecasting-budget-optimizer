import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

PROC_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")

st.set_page_config(page_title="ROAS Forecasting & Budget Allocation", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv(RAW_DIR / "ua_data.csv", parse_dates=["date"])
    holdout = pd.read_csv(PROC_DIR / "holdout_predictions.csv", parse_dates=["date"])
    bt = pd.read_csv(PROC_DIR / "backtest_metrics.csv", parse_dates=["train_end","test_start","test_end"])
    recs = pd.read_csv(PROC_DIR / "recommendations.csv", parse_dates=["date"])
    port_base = pd.read_csv(PROC_DIR / "portfolio_baseline.csv")
    port_after = pd.read_csv(PROC_DIR / "portfolio_after.csv")
    return df, holdout, bt, recs, port_base, port_after

try:
    df, holdout, bt, recs, port_base, port_after = load_data()
except Exception:
    st.error("Run `python main.py` first to generate artifacts.")
    st.stop()

st.title("Mobile Game UA: ROAS Forecasting and Budget Allocation")

# KPIs
col1, col2, col3 = st.columns(3)
with col1:
    mape = bt["MAPE"].mean()
    st.metric("Backtest MAPE (D7 ROAS)", f"{mape*100:.1f}%")
with col2:
    rmse = bt["RMSE"].mean()
    st.metric("Backtest RMSE (D7 ROAS)", f"{rmse:.3f}")
with col3:
    wape = bt["WAPE"].mean()
    st.metric("Backtest WAPE (D7 ROAS)", f"{wape*100:.1f}%")

st.markdown("Historical & Forecasted ROAS (Holdout)")

channel_filter = st.multiselect("Channels", sorted(df["channel"].unique().tolist()), default=None)
campaign_filter = st.multiselect("Campaigns", sorted(df["campaign_id"].unique().tolist()), default=None)

plot_df = holdout.copy()
if channel_filter:
    plot_df = plot_df[plot_df["channel"].isin(channel_filter)]
if campaign_filter:
    plot_df = plot_df[plot_df["campaign_id"].isin(campaign_filter)]

st.line_chart(plot_df.set_index("date")[["ROAS_d7","pred_d7","pred_d7_p05","pred_d7_p95"]])

st.subheader("Scale/Cut Recommendations")
st.dataframe(recs.sort_values(["decision","pred_roas"], ascending=[True, False]))

st.subheader("Budget Reallocation Simulator")
available_channels = sorted(df["channel"].unique().tolist())
src = st.selectbox("Move budget FROM channel", available_channels)
dst = st.selectbox("Move budget TO channel", [c for c in available_channels if c != src])
amount = st.number_input("Amount to reallocate ($)", min_value=100.0, value=10000.0, step=100.0)
min_spend = st.number_input("Minimum spend per campaign ($)", min_value=0.0, value=100.0, step=50.0)
max_shift = st.slider("Max shift ratio per campaign", min_value=0.1, max_value=0.8, value=0.3, step=0.05)

if st.button("Simulate Reallocation"):
    from src.business import budget_reallocation
    portfolio = port_base.copy()
    realloc_map = {f"{src}->{dst}": amount}
    after, new_roas = budget_reallocation(portfolio, realloc_map, min_spend_by_campaign=min_spend, max_shift_ratio=max_shift)
    baseline_roas = (portfolio["current_spend"]*portfolio["pred_roas"]).sum() / portfolio["current_spend"].sum()
    uplift = 100*(new_roas - baseline_roas)/baseline_roas
    st.metric("Portfolio ROAS (baseline)", f"{baseline_roas:.3f}")
    st.metric("Portfolio ROAS (after)", f"{new_roas:.3f}")
    st.metric("Expected uplift", f"{uplift:.2f}%")
    st.dataframe(after.sort_values("channel"))