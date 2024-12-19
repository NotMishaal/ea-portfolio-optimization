import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(file_path, tickers=None):
    """
    Load a CSV file from the given file_path and filter by the given tickers if provided.
    """
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    
    if tickers:
        df = df[df["symbol"].isin(tickers)]
        
    df_pivoted = df.pivot(index="date", columns="symbol", values="close")
    
    # Sort the index and drop any NaN values
    df_pivoted = df_pivoted.sort_index()
    df_pivoted = df_pivoted.fillna(method='ffill').dropna()
    
    return df_pivoted
    
def compute_returns(df):
    return df.pct_change().dropna()

def normalize_returns(df):
    """
    Normalize the returns to have mean 0 and std 1 using StandardScaler.
    """
    scaler = StandardScaler()
    scaled_returns = pd.DataFrame(scaler.fit_transform(df), 
                              index=df.index, 
                              columns=df.columns)
    
    return scaled_returns
