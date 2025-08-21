from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
APP_DIR = Path("app")
SRC_DIR = Path("src")

def ensure_dirs():
    for d in [DATA_DIR, RAW_DIR, PROC_DIR, APP_DIR, SRC_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def save_df(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def load_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])

def set_seed(seed: int = 42):
    np.random.seed(seed)

def add_weekday_weekend(df: pd.DataFrame):
    df["weekday"] = df["date"].dt.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    return df

def add_month_seasonality(df: pd.DataFrame):
    df["month"] = df["date"].dt.month
    return df

def add_holiday_flags(df: pd.DataFrame, country="US"):
    try:
        import holidays
        hol = holidays.country_holidays(country)
        dates = pd.to_datetime(df["date"]).dt.date
        df["is_holiday"] = dates.astype("datetime64").isin(hol).astype(int)
    except Exception:
        df["is_holiday"] = 0
    return df

def clip_nonnegative(df: pd.DataFrame, cols):
    for c in cols:
        df[c] = df[c].clip(lower=0)
    return df
