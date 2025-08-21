import pandas as pd
import numpy as np

def add_lags_and_rolls(df, group_cols=["campaign_id"], target_cols=None, lags=[1,7,14], rolls=[7,14,28]):
    if target_cols is None:
        target_cols = ["ad_spend","installs","revenue_day7","revenue_day30","ROAS_d7","ROAS_d30"]
    df = df.sort_values(["campaign_id","date"]).copy()
    for g in group_cols:
        pass
    df_list = []
    for g, gdf in df.groupby(group_cols, as_index=False):
        gdf = gdf.sort_values("date")
        for col in target_cols:
            for L in lags:
                gdf[f"{col}_lag{L}"] = gdf[col].shift(L)
            for W in rolls:
                gdf[f"{col}_roll{W}"] = gdf[col].rolling(W, min_periods=max(3, int(W*0.4))).mean()
        df_list.append(gdf)
    out = pd.concat(df_list, axis=0)
    return out

def encode_cats(df):
    # keep human-readable categoricals; models can handle category encoding
    cat_cols = ["channel","country","platform","campaign_id"]
    for c in cat_cols:
        df[c] = df[c].astype("category")
    return df

def final_feature_table(df):
    # Example features for tabular model predicting ROAS_d7 and ROAS_d30
    X_cols = [
        "ad_spend","installs","CAC","is_weekend","is_holiday","weekday","month",
        "bid","target_cpi"
    ]
    # Add lag/roll features
    lag_roll_candidates = [c for c in df.columns if any(k in c for k in ["lag","roll"])]
    X_cols.extend(lag_roll_candidates)
    # Include categorical features (will be handled by LightGBM)
    X_cats = ["channel","country","platform","campaign_id"]
    y_cols = ["ROAS_d7","ROAS_d30"]
    # Drop rows with NaNs introduced by lags at the start
    X = df[X_cols + X_cats + ["date"]].copy()
    y = df[y_cols].copy()
    return X, y, X_cats