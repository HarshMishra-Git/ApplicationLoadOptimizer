from prophet import Prophet
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

class Forecaster:
    def __init__(self):
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode='multiplicative'
        )
        self.forecast = None
        
    def train(self, data):
        """Train the Prophet model"""
        self.model.fit(data)
        
    def predict(self, periods=24):
        """Generate forecasts"""
        future = self.model.make_future_dataframe(periods=periods, freq='H')
        self.forecast = self.model.predict(future)
        return self.forecast
        
    def get_metrics(self, actual, predicted):
        """Calculate forecast accuracy metrics"""
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse
        }
        
    def get_components(self):
        """Get trend and seasonality components"""
        return self.model.plot_components(self.forecast)
