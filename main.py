from src.utils import ensure_dirs, RAW_DIR, PROC_DIR, load_df, save_df
from src.data_gen import generate_and_save
from src.features import add_lags_and_rolls, encode_cats, final_feature_table
from src.tabular import train_lgbm_point, train_lgbm_quantile, predict_all
from src.backtest import rolling_origin_backtest
from src.business import rank_scale_cut, project_spend, budget_reallocation

import pandas as pd
import numpy as np

def main():
    ensure_dirs()
    # 1) Data
    try:
        df = load_df(RAW_DIR / "ua_data.csv")
    except Exception:
        df = generate_and_save()

    # 2) Prepare features
    df = df.sort_values(["campaign_id","date"])
    df = add_lags_and_rolls(df)
    df = encode_cats(df)
    X, y, cats = final_feature_table(df)

    # 3) Split train/test by time
    cutoff = df["date"].max() - pd.Timedelta(days=28)
    train_idx = X["date"] <= cutoff
    test_idx = X["date"] > cutoff

    targets = {"ROAS_d7":"pred_d7", "ROAS_d30":"pred_d30"}
    models = {}

    # Train point estimates
    for target, tag in targets.items():
        m = train_lgbm_point(X.loc[train_idx], y.loc[train_idx, target], cats)
        models[f"{tag}_point"] = m

    # Train quantiles for uncertainty
    for target, tag in targets.items():
        m05 = train_lgbm_quantile(X.loc[train_idx], y.loc[train_idx, target], cats, alpha=0.05)
        m95 = train_lgbm_quantile(X.loc[train_idx], y.loc[train_idx, target], cats, alpha=0.95)
        models[f"{tag}_q05"] = m05
        models[f"{tag}_q95"] = m95

    # 4) Evaluate on holdout
    preds_test = {}
    for target, tag in targets.items():
        preds_test[f"{tag}_point"] = models[f"{tag}_point"].predict(X.loc[test_idx].drop(columns=["date"]))
        preds_test[f"{tag}_q05"] = models[f"{tag}_q05"].predict(X.loc[test_idx].drop(columns=["date"]))
        preds_test[f"{tag}_q95"] = models[f"{tag}_q95"].predict(X.loc[test_idx].drop(columns=["date"]))

    eval_df = df.loc[test_idx.values, ["date","campaign_id","channel","ad_spend","ROAS_d7","ROAS_d30"]].copy()
    eval_df["pred_d7"] = preds_test["pred_d7_point"]
    eval_df["pred_d7_p05"] = preds_test["pred_d7_q05"]
    eval_df["pred_d7_p95"] = preds_test["pred_d7_q95"]
    eval_df["pred_d30"] = preds_test["pred_d30_point"]
    eval_df["pred_d30_p05"] = preds_test["pred_d30_q05"]
    eval_df["pred_d30_p95"] = preds_test["pred_d30_q95"]

    # 5) Backtesting (rolling-origin) on D7 ROAS
    feature_cols = [c for c in X.columns if c not in ["date"]]
    bt = rolling_origin_backtest(
        pd.concat([X, y[["ROAS_d7"]]], axis=1),
        date_col="date",
        target_col="ROAS_d7",
        feature_cols=[c for c in feature_cols], 
        cats=cats,
        model_fn=lambda Xtr, ytr, cats: train_lgbm_point(Xtr, ytr, cats),
        n_splits=4,
        min_train_days=180,
        horizon_days=28
    )

    # 6) Business layer on last test day
    last_day = eval_df["date"].max()
    latest = eval_df[eval_df["date"]==last_day].copy()
    latest["pred_roas"] = latest["pred_d7"]  # choose D7 as optimization target
    roas_target = 0.8
    recs = rank_scale_cut(latest, roas_target=roas_target, metric_col="pred_roas", lower_col="pred_d7_p05")

    # Baseline portfolio and budget move example
    portfolio = latest.rename(columns={"ad_spend":"current_spend"})[["campaign_id","channel","current_spend","pred_roas"]].copy()
    realloc_map = {"TikTok->Google": 10000.0}  # demo scenario
    min_spend = 100.0
    port_after, new_roas = budget_reallocation(portfolio, realloc_map, min_spend_by_campaign=min_spend, max_shift_ratio=0.3)
    baseline_roas = (portfolio["current_spend"]*portfolio["pred_roas"]).sum() / portfolio["current_spend"].sum()
    uplift_pct = 100*(new_roas - baseline_roas)/baseline_roas

    # Save artifacts
    from src.utils import save_df, PROC_DIR
    save_df(eval_df, PROC_DIR / "holdout_predictions.csv")
    save_df(bt, PROC_DIR / "backtest_metrics.csv")
    save_df(recs, PROC_DIR / "recommendations.csv")
    save_df(portfolio, PROC_DIR / "portfolio_baseline.csv")
    save_df(port_after, PROC_DIR / "portfolio_after.csv")

    print("Backtest metrics (last split):")
    print(bt.tail(1))
    print(f"Baseline portfolio ROAS: {baseline_roas:.3f}")
    print(f"New portfolio ROAS after reallocation: {new_roas:.3f}")
    print(f"Expected uplift: {uplift_pct:.2f}%")
    print("Artifacts saved in data/processed")

if __name__ == "__main__":
    main()