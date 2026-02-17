
import sys
import os
import pandas as pd
import numpy as np
import time
import ccxt
from datetime import datetime
import csv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import fetch_data, save_data, load_data
from features_volatility import add_volatility_indicators, create_target_barrier, preprocess_data
from models import train_models, get_best_model, save_model, load_model
import config
import recovery # New Module

# --- CONFIGURATION ---
class VolatilityConfig:
    """
    Configuration for High Volatility Trading.
    Mapped to config.py for centralization.
    """
    TIMEFRAME = config.TIMEFRAME
    PROFIT_TARGET = config.TAKE_PROFIT_PCT
    STOP_LOSS = config.STOP_LOSS_PCT
    LOOKAHEAD = 24          # Keep local or move to config if needed (e.g. 6 hours on 15m)
    REQUIRED_PROB = config.BUY_THRESHOLD
    SELL_PROB = config.SELL_THRESHOLD
    CHECK_INTERVAL = config.CHECK_INTERVAL_SECONDS

# Global variable for current symbol
CURRENT_SYMBOL = 'ETH/USDT'

# --- UTILS ---
def get_date_input(prompt):
    while True:
        date_str = input(prompt + " (YYYY-MM-DD) [Enter for None]: ").strip()
        if not date_str:
            return None
        try:
            pd.to_datetime(date_str)
            return date_str
        except ValueError:
            print("Invalid date format. Please try again.")

def filter_by_date(df, start_date, end_date):
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]
    return df

def get_data(symbol=None):
    if symbol is None:
        symbol = CURRENT_SYMBOL
        
    df = load_data(symbol, timeframe=VolatilityConfig.TIMEFRAME)
    if df is None:
        print(f"Data for {symbol} not found locally. Fetching from Binance...")
        start_date = input("Enter start date for data (YYYY-MM-DD): ").strip() or '2023-01-01'
        df = fetch_data(symbol, start_date=start_date, interval=VolatilityConfig.TIMEFRAME)
        if df is not None:
            save_data(df, symbol, timeframe=VolatilityConfig.TIMEFRAME)
    
    if df is not None:
        print("Adding Volatility Indicators...")
        df = add_volatility_indicators(df)
        
        print(f"Creating Barrier Target (P: {VolatilityConfig.PROFIT_TARGET*100}%, S: {VolatilityConfig.STOP_LOSS*100}%)...")
        df = create_target_barrier(df, 
                                   profit_pct=VolatilityConfig.PROFIT_TARGET, 
                                   stop_pct=VolatilityConfig.STOP_LOSS, 
                                   lookahead=VolatilityConfig.LOOKAHEAD)
        
        df = preprocess_data(df)
        print(f"Data ready. Rows: {len(df)}")
    
    return df

# --- WORKFLOWS ---

def train_workflow(df):
    print("\n--- Train High Volatility Model ---")
    start_date = get_date_input("Training Start Date")
    end_date = get_date_input("Training End Date")
    
    train_df = filter_by_date(df, start_date, end_date)
    print(f"Training on {len(train_df)} records.")
    
    # Define Features for High Volatility
    FEATURES_COLUMNS = ['rsi', 'macd', 'macd_signal', 'macd_diff', 
                        'bb_width', 'bb_pband', 
                        'atr_pct', 'adx', 'roc',
                        'sma_20', 'ema_12']

    data = train_df[FEATURES_COLUMNS + ['target']].dropna()
    X = data[FEATURES_COLUMNS]
    y = data['target']
    
    split = int(len(data) * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test, y_test = X.iloc[split:], y.iloc[split:]
    
    print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")
    print(f"Class Distribution (Train): \n{y_train.value_counts()}")
    
    trained_models = train_models(X_train, y_train)
    # Use Accuracy for Multi-Class selection for now
    best_model = get_best_model(trained_models, X_test, y_test, metric='Accuracy')
    
    save = input("Save this model? (y/n): ").lower()
    if save == 'y':
        name = input("Enter model name (e.g. eth_vol_model.pkl): ")
        if not name.endswith('.pkl'): name += '.pkl'
        save_model(best_model, name)
        
    return best_model

def get_features_columns():
    return ['rsi', 'macd', 'macd_signal', 'macd_diff', 
            'bb_width', 'bb_pband', 
            'atr_pct', 'adx', 'roc',
            'sma_20', 'ema_12']

def backtest_workflow(df, model):
    if model is None:
        print("No model loaded.")
        return

    print("\n--- Backtest Barrier Strategy ---")
    start_date = get_date_input("Backtest Start Date")
    end_date = get_date_input("Backtest End Date")
    
    test_df = filter_by_date(df, start_date, end_date).copy()
    print(f"Backtesting on {len(test_df)} records.")
    
    # Check if model has feature names (Scikit-Learn models usually do)
    if hasattr(model, 'feature_names_in_'):
        features = list(model.feature_names_in_)
        print(f"Using model's expected features: {features}")
    elif hasattr(model, 'named_steps') and 'clf' in model.named_steps and hasattr(model.named_steps['clf'], 'feature_names_in_'):
         # Handle Pipeline: Pipeline itself might not have it, but the classifier might
         features = list(model.named_steps['clf'].feature_names_in_)
         print(f"Using pipeline classifier's expected features: {features}")
    else:
        features = get_features_columns()
        print(f"Using default volatility features: {features}")
        
    # Validation
    missing = [f for f in features if f not in test_df.columns]
    if missing:
        print(f"❌ Error: Model requires features {missing} which are not in the data.")
        return

    # Multi-Class Probabilities
    X = test_df[features]
    if hasattr(model, 'predict_proba'):
        all_probs = model.predict_proba(X)
        # Check class indices
        if hasattr(model, 'classes_'):
            classes = list(model.classes_)
            # Default
            col_1 = 1 if 1 in classes else -1
            col_2 = 2 if 2 in classes else -1
            
            prob_long_arr = all_probs[:, classes.index(1)] if col_1 != -1 else np.zeros(len(X))
            prob_short_arr = all_probs[:, classes.index(2)] if col_2 != -1 else np.zeros(len(X))
        else:
            # Assume 0, 1, 2 sorted
            if all_probs.shape[1] >= 3:
                prob_long_arr = all_probs[:, 1]
                prob_short_arr = all_probs[:, 2]
            else:
                 # Fallback for binary model loaded by mistake?
                 prob_long_arr = all_probs[:, 1]
                 prob_short_arr = np.zeros(len(X))
    else:
        print("Model does not support predict_proba.")
        return

    test_df['prob_long'] = prob_long_arr
    test_df['prob_short'] = prob_short_arr
    
    # Simulation Logic
    initial_balance = 1000
    balance = initial_balance
    trades = []
    
    # Pre-calculate Buy & Hold
    if len(test_df) > 0:
        initial_price = test_df['close'].iloc[0]
        final_price = test_df['close'].iloc[-1]
        bh_return = (final_price - initial_price) / initial_price
        bh_equity_curve = [initial_balance * (price / initial_price) for price in test_df['close']]
    else:
        bh_return = 0
        bh_equity_curve = []
    
    equity_curve = [initial_balance]
    dates = [test_df.index[0]]
    
    closes = test_df['close'].values
    highs = test_df['high'].values
    lows = test_df['low'].values
    times = test_df.index
    
    i = 0
    n = len(test_df)
    
    while i < n - 1:
        # LONG Signal
        if prob_long_arr[i] > VolatilityConfig.REQUIRED_PROB:
            entry_price = closes[i]
            tp_price = entry_price * (1 + VolatilityConfig.PROFIT_TARGET)
            sl_price = entry_price * (1 - VolatilityConfig.STOP_LOSS)
            entry_time = times[i]
            
            outcome = 'TIMEOUT'
            exit_price = closes[min(i + VolatilityConfig.LOOKAHEAD, n-1)]
            exit_time = times[min(i + VolatilityConfig.LOOKAHEAD, n-1)]
            idx_exit = min(i + VolatilityConfig.LOOKAHEAD, n-1)
            
            # Check next candles
            for j in range(i + 1, min(i + VolatilityConfig.LOOKAHEAD + 1, n)):
                if lows[j] <= sl_price:
                    outcome = 'STOP_LOSS'
                    exit_price = sl_price
                    exit_time = times[j]
                    idx_exit = j
                    break
                if highs[j] >= tp_price:
                    outcome = 'TAKE_PROFIT'
                    exit_price = tp_price
                    exit_time = times[j]
                    idx_exit = j
                    break
            
            pnl = (exit_price - entry_price) / entry_price
            balance *= (1 + pnl)
            
            trades.append({
                'Entry': entry_time,
                'Exit': exit_time,
                'Type': 'LONG',
                'Outcome': outcome,
                'PnL': pnl,
                'Balance': balance
            })
            
            equity_curve.append(balance)
            dates.append(exit_time)
            i = idx_exit 

        # SHORT Signal (Explicit Class 2 Confidence)
        elif prob_short_arr[i] > VolatilityConfig.REQUIRED_PROB:
            entry_price = closes[i]
            tp_price = entry_price * (1 - VolatilityConfig.PROFIT_TARGET)
            sl_price = entry_price * (1 + VolatilityConfig.STOP_LOSS)
            entry_time = times[i]
            
            outcome = 'TIMEOUT'
            exit_price = closes[min(i + VolatilityConfig.LOOKAHEAD, n-1)]
            exit_time = times[min(i + VolatilityConfig.LOOKAHEAD, n-1)]
            idx_exit = min(i + VolatilityConfig.LOOKAHEAD, n-1)
            
            # Check next candles
            for j in range(i + 1, min(i + VolatilityConfig.LOOKAHEAD + 1, n)):
                if highs[j] >= sl_price:
                    outcome = 'STOP_LOSS'
                    exit_price = sl_price
                    exit_time = times[j]
                    idx_exit = j
                    break
                if lows[j] <= tp_price:
                    outcome = 'TAKE_PROFIT'
                    exit_price = tp_price
                    exit_time = times[j]
                    idx_exit = j
                    break
            
            # PnL for Short: (Entry - Exit) / Entry
            pnl = (entry_price - exit_price) / entry_price
            balance *= (1 + pnl)
            
            trades.append({
                'Entry': entry_time,
                'Exit': exit_time,
                'Type': 'SHORT',
                'Outcome': outcome,
                'PnL': pnl,
                'Balance': balance
            })
            
            equity_curve.append(balance)
            dates.append(exit_time)
            i = idx_exit
        else:
            pass
        
        i += 1
        
    # Results
    print("\n--- Results ---")
    bh_final_balance = initial_balance * (1 + bh_return)
    strat_return = (balance - initial_balance) / initial_balance
    
    print(f"Strategy Final: ${balance:.2f} ({strat_return*100:.2f}%)")
    print(f"Buy & Hold Final: ${bh_final_balance:.2f} ({bh_return*100:.2f}%)")
    
    if trades:
        results = pd.DataFrame(trades)
        win_rate = len(results[results['PnL'] > 0]) / len(results)
        print(f"Win Rate: {win_rate*100:.2f}% | Trades: {len(trades)}")
        print(results['Type'].value_counts())
        
        # Plotting
        try:
            import matplotlib.pyplot as plt
            
            # Create a simplified time series for equity
            # We have points at trade exits.
            equity_df = pd.DataFrame({'Balance': equity_curve}, index=dates)
            equity_df = equity_df.resample('1D').ffill() # Downsample/Upsample for viewing?
            # actually better to just plot the points or reindex to test_df
            
            plt.figure(figsize=(12, 6))
            plt.plot(test_df.index, bh_equity_curve, label='Buy & Hold', linestyle='--', alpha=0.7)
            plt.plot(dates, equity_curve, label='Strategy (ML)', marker='o')
            plt.title(f"Strategy vs Buy & Hold")
            plt.xlabel("Date")
            plt.ylabel("Balance ($)")
            plt.legend()
            plt.grid(True)
            
            save_plot = input("Save plot? (y/n): ").lower()
            if save_plot == 'y':
                fname = f"backtest_results_{int(time.time())}.png"
                plt.savefig(fname)
                print(f"Plot saved to {fname}")
            else:
                plt.show() # Might not work in some envs
                
        except ImportError:
            print("matplotlib not installed. Skipping plot.")
        except Exception as e:
            print(f"Plotting error: {e}")

    else:
        print("No trades triggered.")

# --- LIVE SIMULATION & TRADING ---

def run_live_loop(model, symbol, mode='SIMULATION'):
    """
    Common loop for Real-Time Simulation and Real Trading.
    mode: 'SIMULATION' or 'REAL'
    """
    print(f"\n=== STARTING {mode} MODE for {symbol} ===")
    print(f"Timeframe: {VolatilityConfig.TIMEFRAME}")
    print(f"Target: +{VolatilityConfig.PROFIT_TARGET*100}% | Stop: -{VolatilityConfig.STOP_LOSS*100}%")
    print("Press Ctrl+C to stop.")
    
    # Initialize Exchange (for data fetching and/or trading)
    # Initialize Exchange (for data fetching and/or trading)
    if mode == 'REAL':
        exchange = ccxt.binance({
            'apiKey': config.BINANCE_API_KEY,
            'secret': config.BINANCE_API_SECRET,
            'enableRateLimit': True
        })
    else:
        # Simulation: Use public endpoints only
        exchange = ccxt.binance({
            'enableRateLimit': True
        })
    
    # State
    position = None # { 'entry_price': float, 'quantity': float } (Mock or Real)
    csv_file = f"live_{mode.lower()}_{symbol.replace('/', '_')}.csv"
    
    # Init CSV
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Action', 'Price', 'PnL', 'Quantity', 'Balance', 'Info'])

    def log(action, price, pnl=0, quantity=0, balance=0, info=""):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {action} @ {price} | {info}")
        with open(csv_file, 'a', newline='') as f:
            csv.writer(f).writerow([timestamp, action, price, pnl, quantity, balance, info])

    # --- RECOVERY LOGIC ---
    print("Checking for existing state (Recovery)...")
    recovered_pos, last_bal = recovery.get_last_state(csv_file)
    
    if recovered_pos:
        print(f"⚠️ Recovered Position: {recovered_pos}")
        position = recovered_pos
        # If simulation, maybe restore balance too?
        # In this script, balance is local in 'balance' var but loop resets it?
        # The loop uses 'position' variable.
        
        if mode == 'REAL':
             print("Syncing with Binance...")
             synced_pos, msg = recovery.sync_with_exchange(exchange, symbol, position, mode='SPOT') # Defaulting to SPOT for now as script is mostly spot
             print(f"Sync Result: {msg}")
             position = synced_pos
    else:
        print("No active position found in logs.")

    if last_bal and mode == 'SIMULATION':
         # If we tracked balance in CSV (which we are adding now), we could restore it.
         # But the script doesn't maintain a persistent balance variable across runs easily without reading it.
         # For now, just logging it.
         pass


    try:
        while True:
            # 1. Fetch Data (Enough for indicators)
            ohlcv = exchange.fetch_ohlcv(symbol, VolatilityConfig.TIMEFRAME, limit=100)
            df_live = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_live['timestamp'] = pd.to_datetime(df_live['timestamp'], unit='ms')
            df_live.set_index('timestamp', inplace=True)
            
            df_live = add_volatility_indicators(df_live)
            df_live = preprocess_data(df_live)
            
            # Determine features from model
            if hasattr(model, 'feature_names_in_'):
                features = list(model.feature_names_in_)
            elif hasattr(model, 'named_steps') and 'clf' in model.named_steps and hasattr(model.named_steps['clf'], 'feature_names_in_'):
                features = list(model.named_steps['clf'].feature_names_in_)
            else:
                features = get_features_columns()

            current_price = df_live['close'].iloc[-1]
            
            # Predict
            last_row = df_live.iloc[[-1]][features] # Double brackets to keep DataFrame format
            
            prob_long = 0.0
            prob_short = 0.0
            
            if hasattr(model, 'predict_proba'):
                all_probs = model.predict_proba(last_row)[0]
                if hasattr(model, 'classes_'):
                    classes = list(model.classes_)
                    if 1 in classes: prob_long = all_probs[classes.index(1)]
                    if 2 in classes: prob_short = all_probs[classes.index(2)]
                else:
                     if len(all_probs) >= 3:
                         prob_long = all_probs[1]
                         prob_short = all_probs[2]
                     elif len(all_probs) == 2:
                         prob_long = all_probs[1] 
                         # Binary fallback? No, we retrained. Assume 0.
            else:
                 print("Model has no predict_proba support")
                 
            print(f"Price: {current_price} | Long: {prob_long:.4f} | Short: {prob_short:.4f} | Pos: {'YES' if position else 'NO'}")
            
            # 2. Logic
            
            # EXIT LOGIC (If in position)
            if position:
                side = position.get('side', 'LONG')
                entry = position['entry_price']
                
                # Determine Targets based on Side
                if side == 'LONG':
                    tp_price = entry * (1 + VolatilityConfig.PROFIT_TARGET)
                    sl_price = entry * (1 - VolatilityConfig.STOP_LOSS)
                    
                    # Long Exit Conditions
                    if current_price >= tp_price:
                        action = "SELL" # Close Long
                        reason = "TAKE_PROFIT"
                    elif current_price <= sl_price:
                        action = "SELL" # Close Long
                        reason = "STOP_LOSS"
                    else:
                        action = None
                        
                elif side == 'SHORT':
                    tp_price = entry * (1 - VolatilityConfig.PROFIT_TARGET)
                    sl_price = entry * (1 + VolatilityConfig.STOP_LOSS)
                    
                    # Short Exit Conditions
                    if current_price <= tp_price:
                        action = "BUY" # Close Short
                        reason = "TAKE_PROFIT"
                    elif current_price >= sl_price:
                        action = "BUY" # Close Short
                        reason = "STOP_LOSS"
                    else:
                        action = None

                if action:
                    # Calculate PnL
                    if side == 'LONG':
                        pnl_pct = (current_price - entry) / entry
                    else: # SHORT
                        pnl_pct = (entry - current_price) / entry
                        
                    log(action, current_price, pnl_pct, position['quantity'], 0, f"{reason} ({side})")
                    
                    if mode == 'REAL':
                        # REAL EXECUTION
                        print(f"⚠️ REAL TRADING: WOULD {action} NOW ({reason})")
                        # TODO: Implement real order
                    
                    position = None
            
            # ENTRY LOGIC (If no position)
            elif not position:
                # LONG Entry (Explicit Class 1 Confidence)
                if prob_long > VolatilityConfig.REQUIRED_PROB:
                    qty = 1
                    if mode == 'REAL':
                        print(f"⚠️ REAL TRADING: WOULD BUY (LONG) NOW (Prob {prob_long})")
                        position = {'entry_price': current_price, 'quantity': 0, 'side': 'LONG'}
                    else:
                        position = {'entry_price': current_price, 'quantity': 1, 'side': 'LONG'}
                    
                    log("BUY", current_price, 0, 1, 0, f"OPEN LONG (Prob: {prob_long:.4f})")
                
                # SHORT Entry (Explicit Class 2 Confidence)
                elif prob_short > VolatilityConfig.REQUIRED_PROB:
                    qty = 1
                    if mode == 'REAL':
                         print(f"⚠️ REAL TRADING: WOULD SELL (SHORT) NOW (Prob {prob_short})")
                         position = {'entry_price': current_price, 'quantity': 0, 'side': 'SHORT'}
                    else:
                         position = {'entry_price': current_price, 'quantity': 1, 'side': 'SHORT'}
                         
                    log("SELL", current_price, 0, 1, 0, f"OPEN SHORT (Prob: {prob_short:.4f})")

            # Sleep
            time.sleep(VolatilityConfig.CHECK_INTERVAL)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")


def main():
    print("=== HIGH VOLATILITY TRADING BOT (BARRIER STRATEGY) ===")
    
    global CURRENT_SYMBOL
    sym = input(f"Enter symbol [Default: {CURRENT_SYMBOL}]: ").strip().upper()
    if sym: CURRENT_SYMBOL = sym
    
    df = get_data(CURRENT_SYMBOL)
    current_model = None
    
    while True:
        print(f"\n--- Menu ({CURRENT_SYMBOL}) ---")
        print(f"Model: {'Loaded' if current_model else 'None'}")
        print("1. Train New Model")
        print("2. Load Existing Model")
        print("3. Backtest")
        print("4. Start Real-Time Simulation")
        print("5. Start REAL TRADING (Binance)")
        print("6. Change Config Targets")
        print("7. Exit")
        
        choice = input("Select: ")
        
        if choice == '1':
            current_model = train_workflow(df)
        elif choice == '2':
            # Simple list
            if not os.path.exists('models'): os.makedirs('models')
            files = os.listdir('models')
            for i,f in enumerate(files): print(f"{i}. {f}")
            try:
                idx = int(input("Index: "))
                current_model = load_model(files[idx])
            except: print("Invalid")
        elif choice == '3':
            backtest_workflow(df, current_model)
        elif choice == '4':
            if current_model: run_live_loop(current_model, CURRENT_SYMBOL, 'SIMULATION')
            else: print("Load model first.")
        elif choice == '5':
            if current_model:
                confirm = input("⚠️ REAL TRADING: Are you sure? (yes/no): ")
                if confirm == 'yes':
                    run_live_loop(current_model, CURRENT_SYMBOL, 'REAL')
            else: print("Load model first.")
        elif choice == '6':
            try:
                VolatilityConfig.PROFIT_TARGET = float(input(f"Target (Curr: {VolatilityConfig.PROFIT_TARGET}): "))
                VolatilityConfig.STOP_LOSS = float(input(f"Stop (Curr: {VolatilityConfig.STOP_LOSS}): "))
                print("Config updated.")
            except: print("Invalid")
        elif choice == '7':
            break

if __name__ == "__main__":
    main()
