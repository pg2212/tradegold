import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_backtest(csv_file="gold_predictions.csv", spread_cost=0.50):
    """
    Reads gold_predictions.csv, simulates trade execution on out-of-sample data,
    deducts spread costs, calculates performance metrics, and plots the equity curve.
    """
    print(f"Loading out-of-sample predictions from '{csv_file}'...")
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    
    # 1. Calculate price change per bar
    df['price_change'] = df['Close'].diff()
    
    # 2. Shift position by 1 bar to simulate entering on the NEXT open/close (no lookahead bias)
    df['position'] = df['signal'].shift(1).fillna(0)
    
    # 3. Identify new trade entries to apply spread/slippage costs
    df['trade_entry'] = (df['position'] == 1) & (df['position'].shift(1) == 0)
    
    # 4. Calculate Gross and Net PnL (USD per Ounce)
    df['gross_pnl'] = df['position'] * df['price_change']
    df['costs'] = np.where(df['trade_entry'], spread_cost, 0.0)
    df['net_pnl'] = df['gross_pnl'] - df['costs']
    
    # 5. Cumulative Equity Streams
    df['cum_net_pnl'] = df['net_pnl'].cumsum()
    df['cum_buy_hold'] = df['Close'] - df['Close'].iloc[0]
    
    # 6. Compute Key Performance Indicators (KPIs)
    total_net_pnl = df['cum_net_pnl'].iloc[-1]
    buy_hold_pnl = df['cum_buy_hold'].iloc[-1]
    total_trades = int(df['trade_entry'].sum())
    
    # Calculate Win Rate across closed trades
    winning_trades = (df[df['trade_entry']]['net_pnl'] > 0).sum()
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    
    # Calculate Annualized Sharpe Ratio (assuming 1-hour bars, ~2000 trading hours/yr)
    mean_pnl = df['net_pnl'].mean()
    std_pnl = df['net_pnl'].std()
    sharpe = (mean_pnl / (std_pnl + 1e-9)) * np.sqrt(2000)
    
    print("\n==========================================")
    print("      XAU/USD ML BACKTEST PERFORMANCE     ")
    print("==========================================")
    print(f"Out-of-Sample Period : {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"Total Net Return    : ${total_net_pnl:.2f} / oz (After ${spread_cost} spread/trade)")
    print(f"Buy & Hold Return   : ${buy_hold_pnl:.2f} / oz")
    print(f"Total Trades        : {total_trades}")
    print(f"Win Rate            : {win_rate:.1f}%")
    print(f"Annualized Sharpe   : {sharpe:.2f}")
    print("==========================================\n")
    
    # 7. Plot Performance Graph
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    plt.figure(figsize=(12, 6))
    
    plt.plot(df.index, df['cum_net_pnl'], label="ML Strategy (Net PnL)", color="#1f77b4", linewidth=2)
    plt.plot(df.index, df['cum_buy_hold'], label="Gold Buy & Hold", color="#7f7f7f", linestyle="--", alpha=0.7)
    
    plt.title("XAU/USD Machine Learning Strategy Backtest (Out-of-Sample Test Set)", fontsize=14, pad=15)
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Profit / Loss ($ USD per Ounce)", fontsize=11)
    plt.legend(fontsize=11, loc="upper left")
    plt.tight_layout()
    
    print("Displaying Equity Curve Plot...")
    plt.show()
    
    return df

if __name__ == "__main__":
    results = run_backtest()