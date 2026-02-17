import sys
import os
import pandas as pd
import numpy as np
import random
from datetime import timedelta
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import fetch_data, load_data, save_data
from features import add_technical_indicators, preprocess_data, create_target
from models import load_model, train_models, get_best_model, save_model # Added imports for completeness
from strategy import TradingStrategy
import config

# Configure logging to file/console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def get_data_for_massive_test(symbol):
    """Load or fetch data."""
    print(f"Loading data for {symbol}...")
    df = load_data(symbol)
    if df is None:
        print("Data not found locally. Fetching...")
        df = fetch_data(symbol, start_date='2018-01-01') # Fetch deep history
        if df is not None:
            save_data(df, symbol)
            
    if df is not None:
        # Preprocess once
        print("Generating features...")
        df = add_technical_indicators(df)
        df = preprocess_data(df) # Removes NaNs from indicators
        
    return df

def run_single_simulation(df, model, duration_days, min_start_date=None, initial_capital=10000):
    """
    Run one backtest on a random slice of df.
    """
    # 1. Determine random start point
    # We need at least duration_days of data
    
    total_hours = len(df)
    needed_hours = duration_days * 24
    
    if total_hours <= needed_hours:
        return None
        
    max_start_idx = total_hours - needed_hours
    start_idx = random.randint(0, max_start_idx)
    
    # Slice
    test_df = df.iloc[start_idx : start_idx + needed_hours].copy()
    
    # 2. Run Strategy
    # We use a simplified vectorized backtest for speed
    
    # Prepare Features
    # Check what features model expects
    if hasattr(model, 'feature_names_in_'):
        features = list(model.feature_names_in_)
    else:
        # Fallback to standard features
        features = ['rsi', 'macd', 'macd_signal', 'macd_diff', 
                    'bb_high', 'bb_low', 'bb_mavg', 'sma_20', 'ema_12', 'adx']
                    
    # Predict
    try:
        X = test_df[features]
        # Predict Probabilities if possible
        if hasattr(model, 'predict_proba'):
             probs = model.predict_proba(X)[:, 1]
             # Strategy thresholds
             buy_thresh = 0.55
             sell_thresh = 0.45
             
             # Vectorized Signal
             signals = np.where(probs > buy_thresh, 1, 0)
             signals = np.where(probs < sell_thresh, -1, signals)
        else:
             signals = model.predict(X)
             
        test_df['signal'] = signals
    except Exception as e:
        # If columns missing
        # logger.error(f"Prediction error: {e}")
        return None

    # Calculate Returns
    # hourly_return is return of holding from t to t+1
    test_df['hourly_return'] = test_df['close'].pct_change().shift(-1)
    
    # Strategy Return
    # signal=1 -> Long -> return
    # signal=-1 -> Short -> -return (if we allow shorting? config says spot usually. Let's assume Long-Only vs Buy & Hold for now unless configured)
    # If the user script main.py implies Spot, we assume Long-Only ?
    # The strategy class in src/strategy.py usually returns 1, -1, 0.
    # Let's assume we can go to cash (0) or Long (1).
    # If -1, we sell (Cash).
    
    # Logic:
    # If Signal 1: Held = True
    # If Signal -1: Held = False
    # If Signal 0: Held = Previous Held (Maintain position)
    
    # Vectorized "Position" calculation
    # This is tricky with 0=Hold.
    # Let's assume simplified: 
    # If prob > thresh: Buy (1)
    # If prob < thresh: Sell (0)
    # Else: Hold previous
    
    # Re-calculating position vector
    positions = np.zeros(len(test_df))
    current_pos = 0
    
    sig_arr = test_df['signal'].values
    
    for i in range(len(sig_arr)):
        if sig_arr[i] == 1:
            current_pos = 1
        elif sig_arr[i] == -1:
            current_pos = 0
        positions[i] = current_pos
        
    test_df['position'] = positions
    
    # Strategy Return = Position * Return
    # Shift position? No, if we decide at `t` (close), we enter/exit. 
    # If we enter at Close `t`, we get return of `t` to `t+1`.
    # So Position `t` earns Return `t` (which is change from t to t+1).
    
    test_df['strategy_ret'] = test_df['position'] * test_df['hourly_return']
    test_df['bnh_ret'] = test_df['hourly_return'] # Buy & Hold is always invested (1)
    
    # Cumulative
    total_strat = (1 + test_df['strategy_ret'].fillna(0)).prod() - 1
    total_bnh = (1 + test_df['bnh_ret'].fillna(0)).prod() - 1
    
    final_balance_strat = initial_capital * (1 + total_strat)
    final_balance_bnh = initial_capital * (1 + total_bnh)
    
    return {
        'start_date': test_df.index[0],
        'end_date': test_df.index[-1],
        'strategy_return': total_strat,
        'bnh_return': total_bnh,
        'final_balance_strat': final_balance_strat,
        'final_balance_bnh': final_balance_bnh,
        'pnl_strat': final_balance_strat - initial_capital,
        'pnl_bnh': final_balance_bnh - initial_capital,
        'outperformed': total_strat > total_bnh
    }

def main():
    print("=== MASSIVE BACKTEST SIMULATION ===")
    
    symbol = input("Enter symbol (e.g. BTC/USDT) [Default: BTC/USDT]: ").strip().upper() or 'BTC/USDT'
    
    # List models
    if not os.path.exists('models'):
        print("No models directory found.")
        return
        
    models = [f for f in os.listdir('models') if f.endswith('.pkl')]
    if not models:
        print("No models found.")
        return
        
    print("\nAvailable Models:")
    for i, m in enumerate(models):
        print(f"{i+1}. {m}")
        
    try:
        idx = int(input("Select model: ")) - 1
        model_name = models[idx]
        model = load_model(model_name)
    except:
        print("Invalid selection.")
        return

    try:
        n_sims = int(input("Number of simulations [Default: 100]: ") or 100)
        min_days = int(input("Min duration (days) [Default: 30]: ") or 30)
        max_days = int(input("Max duration (days) [Default: 90]: ") or 90)
    except:
        print("Invalid inputs.")
        return

    # Load Data
    df = get_data_for_massive_test(symbol)
    if df is None:
        return
        
    # Filter by date if requested
    start_date_str = input("Start Date for Tests (YYYY-MM-DD) [Enter for All Data]: ").strip()
    if start_date_str:
        try:
            df = df[df.index >= start_date_str]
            print(f"Filtered data starting from {start_date_str}. Remaining rows: {len(df)}")
            if len(df) < 100: # Basic check
                print("Not enough data after filter.")
                return
        except Exception as e:
            print(f"Error filtering data: {e}")
            return
            
    print(f"\nStarting {n_sims} simulations...")
    print(f"Date Range: {df.index[0]} to {df.index[-1]}")
    
    results = []
    
    for i in range(n_sims):
        # Random duration between min and max
        duration = random.randint(min_days, max_days)
        
        res = run_single_simulation(df, model, duration)
        if res:
            results.append(res)
        else:
            # Retry if simulation failed (e.g. not enough data)
            pass
            
        if (i+1) % 10 == 0:
            print(f"Completed {i+1}/{n_sims} simulations...")
            
    # Analysis
    if not results:
        print("No simulations completed.")
        return
        
    df_res = pd.DataFrame(results)
    
    wins = df_res['outperformed'].sum()
    win_rate = wins / len(df_res)
    
    avg_strat = df_res['strategy_return'].mean()
    avg_bnh = df_res['bnh_return'].mean()
    
    avg_pnl_strat = df_res['pnl_strat'].mean()
    avg_pnl_bnh = df_res['pnl_bnh'].mean()
    
    print("\n" + "="*40)
    print("       RESULTS SUMMARY       ")
    print("="*40)
    print(f"Total Simulations: {len(df_res)}")
    print(f"Model Win Rate vs B&H: {win_rate:.2%}")
    print(f"Avg Strategy Return:   {avg_strat:.2%}")
    print(f"Avg Buy & Hold Return: {avg_bnh:.2%}")
    print(f"Avg Alpha:             {(avg_strat - avg_bnh):.2%}")
    print("-" * 40)
    print(f"Avg PnL (Strategy):    ${avg_pnl_strat:.2f}")
    print(f"Avg PnL (Buy & Hold):  ${avg_pnl_bnh:.2f}")
    print("-" * 40)
    print(f"Best Strategy Run:     {df_res['strategy_return'].max():.2%}")
    print(f"Worst Strategy Run:    {df_res['strategy_return'].min():.2%}")
    print("="*40)
    
    # Save CSV
    df_res.to_csv('massive_backtest_results.csv', index=False)
    print("Detailed results saved to 'massive_backtest_results.csv'")

if __name__ == "__main__":
    main()
