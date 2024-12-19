import pandas as pd
import numpy as np

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

