import numpy as np
import pandas as pd

def rank_scale_cut(forecast_df, roas_target=0.8, metric_col="pred_roas", lower_col=None):
    # forecast_df contains: date, campaign_id, channel, ad_spend (projected), pred_roas, optional lower bound
    df = forecast_df.copy()
    df["decision"] = np.where(df[metric_col] >= roas_target, "SCALE", "CUT")
    if lower_col is not None:
        # risk-aware: require lower bound above a softer threshold
        df["risk_ok"] = df[lower_col] >= (roas_target * 0.7)
        df.loc[~df["risk_ok"], "decision"] = "HOLD"
    df = df.sort_values(metric_col, ascending=False)
    return df

def budget_reallocation(portfolio_df, reallocate_map, min_spend_by_campaign, max_shift_ratio=0.3):
    """
    portfolio_df: rows per campaign with current_spend, pred_roas
    reallocate_map: dict from_channel -> to_channel with amount (total or %); example {"TikTok->Google": 10000}
    min_spend_by_campaign: minimum spend each campaign must retain
    max_shift_ratio: cannot reduce any campaign by more than 30% of its current
    Returns new portfolio_df with updated_spend and expected_portfolio_roas
    """
    df = portfolio_df.copy()
    df["updated_spend"] = df["current_spend"].values

    # Aggregate changes by channel
    for k, amt in reallocate_map.items():
        src, dst = k.split("->")
        amt = float(amt)
        # Remove spend from src campaigns with highest CAC or lowest ROAS first
        src_mask = df["channel"] == src
        src_pool = df.loc[src_mask].sort_values("pred_roas", ascending=True)
        remaining = amt
        for idx, row in src_pool.iterrows():
            cap = min(row["current_spend"]*max_shift_ratio, row["updated_spend"] - min_spend_by_campaign)
            cap = max(cap, 0)
            take = min(cap, remaining)
            df.at[idx, "updated_spend"] = df.at[idx, "updated_spend"] - take
            remaining -= take
            if remaining <= 1e-6:
                break
        # Add to dst campaigns with highest predicted ROAS first
        dst_mask = df["channel"] == dst
        dst_pool = df.loc[dst_mask].sort_values("pred_roas", ascending=False)
        remaining_add = amt
        for idx, row in dst_pool.iterrows():
            df.at[idx, "updated_spend"] = df.at[idx, "updated_spend"] + min(remaining_add, row["current_spend"]*max_shift_ratio)
            remaining_add -= min(remaining_add, row["current_spend"]*max_shift_ratio)
            if remaining_add <= 1e-6:
                break

    # Compute expected portfolio ROAS = sum(updated_spend * pred_roas)/sum(updated_spend)
    df["expected_revenue"] = df["updated_spend"] * df["pred_roas"]
    port_roas = df["expected_revenue"].sum() / max(df["updated_spend"].sum(), 1e-6)
    return df, port_roas

def project_spend(baseline_df, growth_factor=1.0):
    out = baseline_df.copy()
    out["projected_spend"] = out["ad_spend"] * growth_factor
    return out