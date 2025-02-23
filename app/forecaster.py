
from prophet import Prophet
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

class Forecaster:
    def __init__(self):
        self._initialize_model()
        self.forecast = None
        
    def _initialize_model(self):
        """Initialize a new Prophet model"""
        try:
            self.model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                seasonality_mode='multiplicative'
            )
        except Exception as e:
            print(f"Error initializing Prophet model: {e}")
            raise

    def train(self, data):
        """Train the Prophet model"""
        try:
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Input data must be a pandas DataFrame")
            if not all(col in data.columns for col in ['ds', 'y']):
                raise ValueError("Data must contain 'ds' and 'y' columns")
            
            # Initialize new model before training
            self._initialize_model()
            self.model.fit(data)
            return True
        except Exception as e:
            print(f"Error training model: {e}")
            raise

    def predict(self, periods=24):
        """Generate forecasts"""
        try:
            future = self.model.make_future_dataframe(periods=periods, freq='h')
            self.forecast = self.model.predict(future)
            return self.forecast
        except Exception as e:
            print(f"Error generating forecast: {e}")
            raise

    def get_metrics(self, actual, predicted):
        """Calculate forecast accuracy metrics"""
        try:
            mae = mean_absolute_error(actual, predicted)
            mse = mean_squared_error(actual, predicted)
            rmse = np.sqrt(mse)

            return {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse
            }
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            raise

    def get_components(self):
        """Get trend and seasonality components"""
        try:
            if self.forecast is None:
                raise ValueError("Must run predict() before getting components")
            return self.model.plot_components(self.forecast)
        except Exception as e:
            print(f"Error getting components: {e}")
            raise
            raise

# IF YOU WANT TO ADD THE LSTM OR SARIMA MODEL INSTEAD OF PROPHET APPLY FOLLWING CODE

'''


import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

class Forecaster:
    def __init__(self, model_type='sarima'):
        self.model_type = model_type
        self.model = None
        self.scaler = MinMaxScaler()
        self.forecast = None
        
    def prepare_lstm_data(self, data, lookback=24):
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        X, y = [], []
        for i in range(len(scaled_data) - lookback):
            X.append(scaled_data[i:(i + lookback), 0])
            y.append(scaled_data[i + lookback, 0])
        return np.array(X), np.array(y)
        
    def create_lstm_model(self, lookback):
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(lookback, 1), return_sequences=True),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
        
    def train(self, data):
        try:
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Input data must be a pandas DataFrame")
                
            y = data['y'].values
            
            if self.model_type == 'sarima':
                self.model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
                self.model = self.model.fit(disp=False)
            else:  # LSTM
                X, y = self.prepare_lstm_data(y)
                X = X.reshape((X.shape[0], X.shape[1], 1))
                self.model = self.create_lstm_model(24)
                self.model.fit(X, y, epochs=50, batch_size=32, verbose=0)
            
            return True
            
        except Exception as e:
            print(f"Error training model: {e}")
            raise
            
    def predict(self, periods=24):
        try:
            if self.model_type == 'sarima':
                self.forecast = self.model.forecast(periods)
                forecast_df = pd.DataFrame({
                    'ds': pd.date_range(start=pd.Timestamp.now(), periods=periods, freq='H'),
                    'yhat': self.forecast,
                    'yhat_lower': self.forecast - 1.96 * self.model.params_std,
                    'yhat_upper': self.forecast + 1.96 * self.model.params_std
                })
            else:  # LSTM
                last_sequence = self.scaler.transform(self.model.history.history['loss'][-24:].reshape(-1, 1))
                forecast = []
                
                for _ in range(periods):
                    next_pred = self.model.predict(last_sequence.reshape(1, 24, 1), verbose=0)
                    forecast.append(next_pred[0, 0])
                    last_sequence = np.roll(last_sequence, -1)
                    last_sequence[-1] = next_pred
                    
                forecast = self.scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
                forecast_df = pd.DataFrame({
                    'ds': pd.date_range(start=pd.Timestamp.now(), periods=periods, freq='H'),
                    'yhat': forecast.flatten(),
                    'yhat_lower': forecast.flatten() * 0.9,
                    'yhat_upper': forecast.flatten() * 1.1
                })
                
            return forecast_df
            
        except Exception as e:
            print(f"Error generating forecast: {e}")
            raise
            
    def get_metrics(self, actual, predicted):
        try:
            mae = mean_absolute_error(actual, predicted)
            mse = mean_squared_error(actual, predicted)
            rmse = np.sqrt(mse)
            
            return {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse
            }
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            raise
'''
