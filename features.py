import pandas as pd
import numpy as np

def add_features(csv_file="gold_data.csv", output_file="gold_features.csv"):
    """
    Reads raw gold OHLCV data from CSV and calculates technical indicators
    to be used as inputs for Machine Learning models.
    """
    print(f"Reading market data from '{csv_file}'...")
    
    # 1. Load raw data and set Datetime index
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    data = df.copy()
    
    # 2. Log Returns (stationary price change metric)
    data['log_ret'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # 3. Moving Average Ratios (Trend/Momentum)
    data['ema_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['ema_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['ema_ratio'] = data['ema_20'] / data['ema_50']
    
    # 4. Relative Strength Index (RSI - 14 period)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # 5. Average True Range (ATR - 14 period for Dynamic Target/Volatility)
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['atr'] = tr.rolling(window=14).mean()
    
    # 6. Historical Volatility (20-period standard deviation of log returns)
    data['volatility'] = data['log_ret'].rolling(window=20).std()
    
    # Drop initial NaN rows created by rolling windows (e.g., first 20 bars)
    feature_df = data.dropna()
    
    # Save engineered dataset to local CSV file
    feature_df.to_csv(output_file)
    print(f"Features successfully generated and saved to '{output_file}'!")
    print(f"Total valid bars ready for ML processing: {len(feature_df)}")
    
    return feature_df

if __name__ == "__main__":
    features_df = add_features()
    
    print("\nFirst 5 rows of engineered features:")
    print(features_df[['Close', 'log_ret', 'ema_ratio', 'rsi', 'atr', 'volatility']].head())