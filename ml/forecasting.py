import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def calculate_forecast(returns, window_size=20):
    """
    Calculate a simple forecast model as an mvp: next-day expected return as average of past N days,
    and volatility as standard deviation of the same period.
    
    Parameters:
    returns (pd.Series): Series of returns.
    window_size (int): Window size for forecasting. Default is 20.
    
    Returns:
    tuple: Forecasted return and forecasted volatility.
    """
    forecasted_returns = returns.rolling(window=window_size).mean().iloc[-1]
    forecasted_volatility = returns.rolling(window=window_size).std().iloc[-1]
    return forecasted_returns, forecasted_volatility


def forecast_returns_arima(returns, forecast_horizon=1):
    expected_returns = []
    
    for stock in returns.columns:
        series = returns[stock].dropna()
        model = ARIMA(series, order=(1,0,0))
        model_fit = model.fit()
        # Forecast the next step
        forecast = model_fit.forecast(steps=forecast_horizon)
        expected_returns.append(forecast.iloc[-1])  # Use the last forecasted value
    
    return np.array(expected_returns)

def compute_covariance_matrix(returns):
    # Compute covariance matrix from historical returns
    return returns.cov()