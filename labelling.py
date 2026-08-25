import pandas as pd
import numpy as np

def apply_triple_barrier(csv_file="gold_features.csv", output_file="gold_labeled.csv", 
                         pt_multiplier=1.5, sl_multiplier=1.0, max_holding=12):
    """
    Reads gold_features.csv and assigns a binary target to each bar:
     1 -> Take Profit target hit first (Profitable BUY opportunity)
     0 -> Stop Loss or Time Limit reached first (NO BUY / HOLD)
    """
    print(f"Reading engineered features from '{csv_file}'...")
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    
    prices = df['Close'].values
    atrs = df['atr'].values
    n = len(prices)
    labels = []
    
    print("Calculating Triple Barrier targets across historical price bars...")
    for i in range(n):
        # If near the end of dataset, mark as 0 (cannot verify 12 bars out)
        if i + max_holding >= n:
            labels.append(0)
            continue
            
        entry_price = prices[i]
        pt = entry_price + (atrs[i] * pt_multiplier)
        sl = entry_price - (atrs[i] * sl_multiplier)
        
        label = 0  # Default to 0
        for j in range(1, max_holding + 1):
            future_price = prices[i + j]
            if future_price >= pt:
                label = 1  # Profit target hit first
                break
            elif future_price <= sl:
                label = 0  # Stop loss hit first
                break
                
        labels.append(label)
        
    df['target'] = labels
    
    # Save labeled dataset to local CSV
    df.to_csv(output_file)
    
    buy_signals = df['target'].sum()
    total_bars = len(df)
    print(f"\nLabeling complete! Saved to '{output_file}'.")
    print(f"Total Bars Processed: {total_bars}")
    print(f"Buy Opportunities (Target = 1): {buy_signals} ({buy_signals/total_bars*100:.1f}%)")
    print(f"No-Buy / Hold Signals (Target = 0): {total_bars - buy_signals} ({(total_bars - buy_signals)/total_bars*100:.1f}%)")
    
    return df

if __name__ == "__main__":
    labeled_df = apply_triple_barrier()
    
    print("\nTarget preview (First 10 rows):")
    print(labeled_df[['Close', 'atr', 'target']].head(10))