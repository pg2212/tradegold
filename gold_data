import yfinance as yf
import pandas as pd
import numpy as np

def get_gold_data(ticker="GC=F", period="2y", interval="1h"):
    """
    Downloads historical gold market data via Yahoo Finance
    and saves it to a local CSV file.
    """
    print(f"Fetching {ticker} data for period '{period}' with interval '{interval}'...")
    
    # Download data
    df = yf.download(tickers=ticker, period=period, interval=interval)
    
    # Clean multi-index columns if returned by yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # Select required columns and drop missing rows
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df.index = pd.to_datetime(df.index)
    
    # Save to local CSV file inside your VS Code project folder
    csv_filename = "gold_data.csv"
    df.to_csv(csv_filename)
    print(f"Data saved locally to '{csv_filename}'.")
    
    return df

if __name__ == "__main__":
    gold_df = get_gold_data(ticker="GC=F", period="2y", interval="1h")
    
    print(f"\nSuccessfully loaded {len(gold_df)} price bars!")
    print("\nFirst 5 rows:")
    print(gold_df.head())