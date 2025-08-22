import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import xgboost as xgb
from prophet import Prophet
import holidays
import pickle
import os

class ROASPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.results = {}

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {self.df.shape}")
        return self.df

    def prepare_data(self):
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['CAC'] = self.df.apply(lambda x: x['spend'] / x['installs'] if x['installs'] > 0 else 0, axis=1)
        self.df['is_weekend'] = self.df['date'].dt.dayofweek >= 5
        us_holidays = holidays.US()
        self.df['is_holiday'] = self.df['date'].apply(lambda x: x in us_holidays)
        self.df['campaign_encoded'] = self.df['campaign'].astype('category').cat.codes
        self.df['channel_encoded'] = self.df['channel'].astype('category').cat.codes
        self.df['spend_per_install'] = self.df['spend'] / (self.df['installs'] + 1)
        self.df['conversion_rate'] = self.df['conversions'] / (self.df['installs'] + 1)
        self.df['revenue_per_conversion'] = self.df['revenue'] / (self.df['conversions'] + 1)
        print("Data preparation completed.")
        return self.df

    def train_xgboost_models(self):
        targets = ['roas_day_1', 'roas_day_7', 'roas_day_30']
        feature_cols = [
            'spend', 'installs', 'conversions', 'CAC', 'is_weekend', 'is_holiday',
            'campaign_encoded', 'channel_encoded', 'spend_per_install', 'conversion_rate', 'revenue_per_conversion'
        ]
        X = self.df[feature_cols]
        for target in targets:
            print(f"Training model for {target}...")
            y = self.df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            self.models[target] = model
            self.results[target] = {'MAPE': mape, 'RMSE': rmse}
            print(f"{target} - MAPE: {mape:.4f}, RMSE: {rmse:.4f}")
        return self.results

    def prophet_forecast(self, campaign=None, channel=None, periods=30):
        df = self.df.copy()
        if campaign is not None:
            df = df[df['campaign'] == campaign]
        if channel is not None:
            df = df[df['channel'] == channel]
        ts_data = df.groupby('date').agg({'revenue': 'sum'}).reset_index().rename(columns={'date':'ds','revenue':'y'})
        model = Prophet()
        model.add_country_holidays(country_name='US')
        model.fit(ts_data)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        return forecast[['ds','yhat','yhat_lower','yhat_upper']]

    def budget_reallocation_simulator(self, adjustments):
        df = self.df.copy()
        for channel_name, shift in adjustments.items():
            mask = df['channel'] == channel_name
            df.loc[mask, 'spend'] *= (1 + shift)
        df['CAC'] = df.apply(lambda x: x['spend'] / x['installs'] if x['installs'] > 0 else 0, axis=1)
        feature_cols = [
            'spend', 'installs', 'conversions', 'CAC', 'is_weekend', 'is_holiday',
            'campaign_encoded', 'channel_encoded', 'spend_per_install', 'conversion_rate', 'revenue_per_conversion'
        ]
        X_sim = df[feature_cols]
        original_roas = (self.df['roas_day_1'] * self.df['spend']).sum() / self.df['spend'].sum()
        predicted_roas = self.models['roas_day_1'].predict(X_sim)
        new_portfolio_roas = (predicted_roas * df['spend']).sum() / df['spend'].sum()
        uplift = new_portfolio_roas - original_roas
        return {'original_roas': original_roas, 'new_roas': new_portfolio_roas, 'uplift': uplift}

    def get_campaign_recommendations(self, top_n=5):
        feature_cols = [
            'spend', 'installs', 'conversions', 'CAC', 'is_weekend', 'is_holiday',
            'campaign_encoded', 'channel_encoded', 'spend_per_install', 'conversion_rate', 'revenue_per_conversion'
        ]
        X = self.df[feature_cols]
        predicted_roas = self.models['roas_day_1'].predict(X)
        df_recommend = self.df.copy()
        df_recommend['predicted_roas'] = predicted_roas
        top_campaigns = df_recommend.sort_values('predicted_roas', ascending=False).groupby('campaign').first().reset_index()
        return top_campaigns[['campaign', 'predicted_roas']].head(top_n)

    def save_models(self, models_dir='models'):
        os.makedirs(models_dir, exist_ok=True)
        for target, model in self.models.items():
            with open(f"{models_dir}/{target}_model.pkl", "wb") as f:
                pickle.dump(model, f)
        with open(f"{models_dir}/results.pkl", "wb") as f:
            pickle.dump(self.results, f)
        print(f"Models saved in '{models_dir}'")

def main():
    pipeline = ROASPipeline(r'D:\Internship\roas\marketing_campaign_dataset.csv')
    pipeline.load_data()
    pipeline.prepare_data()
    pipeline.train_xgboost_models()
    pipeline.save_models()
    top_campaigns = pipeline.get_campaign_recommendations(top_n=5)
    print("Top Campaign Recommendations:\n", top_campaigns)
    forecast = pipeline.prophet_forecast(periods=30)
    print("Prophet Forecast (last few days):\n", forecast.tail())
    simulation = pipeline.budget_reallocation_simulator({'Facebook': .1, 'Google': -.05})
    print("Budget Reallocation Impact:\n", simulation)

if __name__ == '__main__':
    main()
