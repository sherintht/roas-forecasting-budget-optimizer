import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

def wape(y_true, y_pred):
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return np.nan
    return np.sum(np.abs(y_true - y_pred)) / denom

def rolling_origin_backtest(df, date_col, target_col, feature_cols, cats, model_fn, n_splits=4, min_train_days=120, horizon_days=28):
    # df must be time-sorted
    df = df.sort_values(date_col).copy()
    dates = df[date_col].sort_values().unique()
    start_idx = min_train_days
    step = horizon_days
    results = []

    while start_idx + horizon_days <= len(dates):
        train_end_date = dates[start_idx-1]
        test_dates = dates[start_idx:start_idx+horizon_days]
        train = df[df[date_col] <= train_end_date]
        test = df[df[date_col].isin(test_dates)]
        X_train = train[feature_cols + ["date"]]
        y_train = train[target_col]
        X_test = test[feature_cols + ["date"]]
        y_test = test[target_col]
        model = model_fn(X_train, y_train, cats)
        y_pred = model.predict(X_test.drop(columns=["date"]))
        mape = mean_absolute_percentage_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        w = wape(y_test.values, y_pred)
        results.append({
            "train_end": train_end_date,
            "test_start": test_dates[0],
            "test_end": test_dates[-1],
            "MAPE": mape,
            "RMSE": rmse,
            "WAPE": w
        })
        start_idx += step
    return pd.DataFrame(results)