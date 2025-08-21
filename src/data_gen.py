import numpy as np
import pandas as pd
from .utils import ensure_dirs, save_df, RAW_DIR, set_seed, add_weekday_weekend, add_month_seasonality, add_holiday_flags, clip_nonnegative

CHANNELS = ["Facebook", "Google", "TikTok", "UnityAds", "Applovin"]
COUNTRIES = ["US", "GB", "DE", "JP", "BR"]
PLATFORMS = ["iOS", "Android"]

def generate_campaigns(n_campaigns=20, channels=CHANNELS):
    campaigns = []
    for i in range(n_campaigns):
        channel = np.random.choice(channels)
        country = np.random.choice(COUNTRIES, p=[0.35, 0.15, 0.15, 0.2, 0.15])
        platform = np.random.choice(PLATFORMS, p=[0.45, 0.55])
        base_cpi = np.random.uniform(0.8, 4.0) * (1.15 if platform == "iOS" else 0.95)
        quality = np.random.uniform(0.7, 1.3)  # impacts LTV
        bid = np.random.uniform(0.7, 3.0)
        target_cpi = base_cpi * np.random.uniform(0.9, 1.2)
        campaigns.append({
            "campaign_id": f"C{i+1:03d}",
            "channel": channel,
            "country": country,
            "platform": platform,
            "base_cpi": base_cpi,
            "quality": quality,
            "bid": bid,
            "target_cpi": target_cpi
        })
    return pd.DataFrame(campaigns)

def seasonality_factor(date):
    # weekday/weekend + rough monthly patterns
    weekend_boost = 1.08 if date.weekday() in [5, 6] else 1.0
    month_mult = {
        1: 0.92, 2: 0.95, 3: 1.0, 4: 1.02, 5: 1.05, 6: 1.07, 7: 1.10, 8: 1.08, 9: 1.03, 10: 1.04, 11: 1.12, 12: 1.20
    }.get(date.month, 1.0)
    return weekend_boost * month_mult

def channel_spend_profile(channel):
    return {
        "Facebook": np.random.uniform(800, 2000),
        "Google": np.random.uniform(700, 1800),
        "TikTok": np.random.uniform(400, 1200),
        "UnityAds": np.random.uniform(200, 800),
        "Applovin": np.random.uniform(200, 700)
    }[channel]

def generate_timeseries(campaigns, start_date="2024-01-01", end_date="2025-07-31"):
    dates = pd.date_range(start_date, end_date, freq="D")
    rows = []
    for _, c in campaigns.iterrows():
        base_spend = channel_spend_profile(c["channel"]) * np.random.uniform(0.7, 1.3)
        spend_vol = np.random.uniform(0.15, 0.35)
        ltv_d1 = np.random.uniform(0.15, 0.45) * c["quality"]
        ltv_d7 = ltv_d1 * np.random.uniform(1.8, 3.5)
        ltv_d30 = ltv_d7 * np.random.uniform(1.2, 2.0)
        for d in dates:
            seas = seasonality_factor(d)
            spend = base_spend * seas * np.random.lognormal(mean=0, sigma=spend_vol)
            cpi = c["base_cpi"] * np.random.lognormal(mean=0, sigma=0.25)
            installs = spend / max(cpi, 0.2)
            sat = 1 - np.exp(-spend / 2500)
            d1_revenue = installs * ltv_d1 * (0.9 + 0.2*np.random.rand()) * (0.85 + 0.3*sat)
            d7_revenue = installs * ltv_d7 * (0.9 + 0.2*np.random.rand()) * (0.85 + 0.3*sat)
            d30_revenue = installs * ltv_d30 * (0.9 + 0.2*np.random.rand()) * (0.85 + 0.3*sat)
            rows.append({
               
