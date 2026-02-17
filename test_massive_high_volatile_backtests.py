import sys
import os
import pandas as pd
import numpy as np
import random
import logging
from datetime import timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import fetch_data, load_data, save_data
from features_volatility import add_volatility_indicators, create_target_barrier, preprocess_data
from models import load_model, train_models, get_best_model, save_model
import config

# Configuration mimicking test_high_volatile_trading.py
class VolatilityConfig:
    TIMEFRAME = '15m'       
    PROFIT_TARGET = 0.015   # 1.5%
    STOP_LOSS = 0.010       # 1.0%
    LOOKAHEAD = 24          # 6 hours
    REQUIRED_PROB = 0.60    

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def get_data_for_massive_test(symbol):
    print(f"Loading data for {symbol}...")
    df = load_data(symbol, timeframe=VolatilityConfig.TIMEFRAME)
    if df is None:
        print("Data not found locally. Fetching...")
        df = fetch_data(symbol, start_date='2020-01-01', interval=VolatilityConfig.TIMEFRAME)
        if df is not None:
            save_data(df, symbol, timeframe=VolatilityConfig.TIMEFRAME)
            
    if df is not None:
        print("Generating Volatility features...")
        df = add_volatility_indicators(df)
        df = preprocess_data(df)
        
    return df

def run_barrier_simulation(df, model, duration_days, initial_capital=10000):
    """
    Run barrier strategy backtest on random slice.
    """
    # 1. Slice
    total_samples = len(df)
    # 15m intervals per day = 4 * 24 = 96
    samples_per_day = 96
    needed_samples = duration_days * samples_per_day
    
    if total_samples <= needed_samples:
        return None
        
    max_start = total_samples - needed_samples
    start_idx = random.randint(0, max_start)
    end_idx = start_idx + needed_samples
    
    test_df = df.iloc[start_idx : end_idx].copy()
    
    # 2. Features & Prediction
    features = ['rsi', 'macd', 'macd_signal', 'macd_diff', 
                'bb_width', 'bb_pband', 
                'atr_pct', 'adx', 'roc',
                'sma_20', 'ema_12']
    
    # Check if model has specific features
    if hasattr(model, 'feature_names_in_'):
        features = list(model.feature_names_in_)
        
    try:
        X = test_df[features]
        probs = model.predict_proba(X)[:, 1]
    except Exception as e:
        # logger.error(f"Pred Error: {e}")
        return None
    
    # 3. Simulation Loop (Event Driven for Barriers)
    balance = initial_capital
    initial_price = test_df['close'].iloc[0]
    final_price = test_df['close'].iloc[-1]
    
    trades = 0
    wins = 0
    
    # Arrays for fast access
    closes = test_df['close'].values
    highs = test_df['high'].values
    lows = test_df['low'].values
    times = test_df.index
    n = len(test_df)
    
    i = 0
    while i < n - 1:
        prob = probs[i]
        
        if prob > VolatilityConfig.REQUIRED_PROB:
            # ENTRY LONG
            entry_price = closes[i]
            tp_price = entry_price * (1 + VolatilityConfig.PROFIT_TARGET)
            sl_price = entry_price * (1 - VolatilityConfig.STOP_LOSS)
            
            # Find Exit
            exit_price = closes[min(i + VolatilityConfig.LOOKAHEAD, n-1)] # Default Timeout
            idx_exit = min(i + VolatilityConfig.LOOKAHEAD, n-1)
            outcome = 'TIMEOUT'
            
            for j in range(i + 1, min(i + VolatilityConfig.LOOKAHEAD + 1, n)):
                if lows[j] <= sl_price:
                    outcome = 'STOP_LOSS'
                    exit_price = sl_price
                    idx_exit = j
                    break
                if highs[j] >= tp_price:
                    outcome = 'TAKE_PROFIT'
                    exit_price = tp_price
                    idx_exit = j
                    break
            
            # Calculate Result
            pnl_pct = (exit_price - entry_price) / entry_price
            balance *= (1 + pnl_pct)
            
            trades += 1
            if pnl_pct > 0: wins += 1
            
            # Skip to exit
            i = idx_exit
        else:
            i += 1
            
    # Buy & Hold Result
    bnh_return = (final_price - initial_price) / initial_price
    final_bnh = initial_capital * (1 + bnh_return)
    
    strat_return = (balance - initial_capital) / initial_capital
    
    return {
        'start_date': test_df.index[0],
        'end_date': test_df.index[-1],
        'strategy_return': strat_return,
        'bnh_return': bnh_return,
        'final_balance_strat': balance,
        'final_balance_bnh': final_bnh,
        'pnl_strat': balance - initial_capital,
        'pnl_bnh': final_bnh - initial_capital,
        'strat_win_rate': (wins/trades) if trades > 0 else 0,
        'trades': trades,
        'outperformed': strat_return > bnh_return
    }

def main():
    print("=== MASSIVE HIGH VOLATILITY BACKTEST ===")
    print(f"Strategy: Barrier (TP: {VolatilityConfig.PROFIT_TARGET*100}%, SL: {VolatilityConfig.STOP_LOSS*100}%)")
    
    symbol = input("Enter symbol (e.g. ETH/USDT) [Default: ETH/USDT]: ").strip().upper() or 'ETH/USDT'
    
    # Models
    if not os.path.exists('models'):
        print("No models/")
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
        model = load_model(models[idx])
    except:
        print("Invalid.")
        return

    try:
        n_sims = int(input("Simulations [100]: ") or 100)
        min_days = int(input("Min Days [10]: ") or 10)
        max_days = int(input("Max Days [60]: ") or 60)
    except:
        print("Invalid.")
        return

    # Load
    df = get_data_for_massive_test(symbol)
    if df is None: return

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
    
    print(f"\nRunning {n_sims} simulations...")
    results = []
    
    for i in range(n_sims):
        dur = random.randint(min_days, max_days)
        res = run_barrier_simulation(df, model, dur)
        if res:
            results.append(res)
            
        if (i+1)%10 == 0: print(f"{i+1}/{n_sims}...")
        
    if not results:
        print("No results.")
        return
        
    df_res = pd.DataFrame(results)
    
    # Stats
    win_rate = df_res['outperformed'].mean()
    avg_strat_ret = df_res['strategy_return'].mean()
    avg_bnh_ret = df_res['bnh_return'].mean()
    avg_strat_pnl = df_res['pnl_strat'].mean()
    avg_bnh_pnl = df_res['pnl_bnh'].mean()
    avg_trades = df_res['trades'].mean()
    avg_trade_win = df_res['strat_win_rate'].mean()
    
    print("\n" + "="*45)
    print("       HIGH VOLATILITY RESULTS       ")
    print("="*45)
    print(f"Simulations: {len(df_res)}")
    print(f"Win Rate (Strategy vs B&H): {win_rate:.2%}")
    print(f"Avg Trades per Run:         {avg_trades:.1f}")
    print(f"Avg Trade Win Rate:         {avg_trade_win:.2%}")
    print("-" * 45)
    print(f"Avg Return (Strategy):      {avg_strat_ret:.2%}")
    print(f"Avg Return (Buy & Hold):    {avg_bnh_ret:.2%}")
    print(f"Avg Alpha:                  {(avg_strat_ret - avg_bnh_ret):.2%}")
    print("-" * 45)
    print(f"Avg PnL (Strategy):         ${avg_strat_pnl:.2f}")
    print(f"Avg PnL (Buy & Hold):       ${avg_bnh_pnl:.2f}")
    print("="*45)
    
    df_res.to_csv('massive_hv_results.csv', index=False)
    print("Saved to massive_hv_results.csv")

if __name__ == "__main__":
    main()
