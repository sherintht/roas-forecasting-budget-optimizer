import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor

def train_lgbm_quantile(X, y, cats, alpha, seed=42):
    params = dict(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=seed,
        objective="quantile",
        alpha=alpha
    )
    model = LGBMRegressor(**params)
    # LightGBM can take categorical features by name
    model.fit(
        X.drop(columns=["date"]),
        y,
        categorical_feature=cats,
        verbose=False
    )
    return model

def train_lgbm_point(X, y, cats, seed=42):
    model = LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=seed,
        objective="regression"
    )
    model.fit(
        X.drop(columns=["date"]),
        y,
        categorical_feature=cats,
        verbose=False
    )
    return model

def predict_all(models, X, cats):
    X_in = X.drop(columns=["date"])
    preds = {}
    for k, m in models.items():
        preds[k] = m.predict(X_in)
    return preds