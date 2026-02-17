import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import fetch_data, save_data, load_data
from features import add_technical_indicators, create_target, preprocess_data
from models import train_models, get_best_model, save_model, load_model
from strategy import TradingStrategy
from backtest import Backtester
from simulator import RealTimeSimulator
from real_trading import RealTradingBot
import config

# Global variable for current symbol
CURRENT_SYMBOL = 'BTC/USDT'
DATA_FILE_PREFIX = 'btc_1h' # Legacy, will use get_filename

def get_data(symbol=None):
    if symbol is None:
        symbol = CURRENT_SYMBOL
        
    df = load_data(symbol)
    if df is None:
        print(f"Data for {symbol} not found locally. Fetching from Binance...")
        start_date = input("Enter start date for data fetching (YYYY-MM-DD) [Default: 2020-01-01]: ").strip()
        if not start_date:
            start_date = '2020-01-01'
            
        df = fetch_data(symbol, start_date=start_date)
        if df is not None:
            save_data(df, symbol)
    
    if df is not None:
        # Preprocess immediately
        df = add_technical_indicators(df)
        df = create_target(df)
        df = preprocess_data(df)
    
    return df

def change_symbol():
    global CURRENT_SYMBOL
    symbol = input("Enter symbol (e.g. ETH/USDT, SOL/USDT): ").strip().upper()
    if symbol:
        CURRENT_SYMBOL = symbol
        print(f"Symbol changed to {CURRENT_SYMBOL}")
        return get_data(CURRENT_SYMBOL)
    return None

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

def train_workflow(df):
    print("\n--- Train New Model ---")
    start_date = get_date_input("Training Start Date")
    end_date = get_date_input("Training End Date")
    
    train_df = filter_by_date(df, start_date, end_date)
    print(f"Training on {len(train_df)} records.")
    
    if len(train_df) < 100:
        print("Not enough data to train. Aborting.")
        return None

    # Features columns should be global or passed
    # FEATURES_COLUMNS is defined at top of file
    FEATURES_COLUMNS = ['rsi', 'macd', 'macd_signal', 'macd_diff', 
                        'bb_high', 'bb_low', 'bb_mavg', 'sma_20', 'ema_12', 'adx']

    X = train_df[FEATURES_COLUMNS]
    y = train_df['target']
    
    split = int(len(train_df) * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test, y_test = X.iloc[split:], y.iloc[split:]
    
    trained_models = train_models(X_train, y_train)
    best_model = get_best_model(trained_models, X_test, y_test, metric='Precision')
    
    save = input("Save this model? (y/n): ").lower()
    if save == 'y':
        name = input("Enter model name (e.g. my_model.pkl): ")
        if not name.endswith('.pkl'):
            name += '.pkl'
        save_model(best_model, name)
        
    return best_model

def load_workflow():
    print("\n--- Load Model ---")
    if not os.path.exists('models'):
        print("No models directory found.")
        return None
        
    files = [f for f in os.listdir('models') if f.endswith('.pkl')]
    if not files:
        print("No models found.")
        return None
        
    for i, f in enumerate(files):
        print(f"{i+1}. {f}")
        
    try:
        choice = int(input("Select model number: ")) - 1
        if 0 <= choice < len(files):
            return load_model(files[choice])
        else:
            print("Invalid selection.")
    except ValueError:
        print("Invalid input.")
    return None

def backtest_workflow(df, model):
    if model is None:
        print("No model loaded. Please train or load a model first.")
        return

    print("\n--- Backtest Model ---")
    start_date = get_date_input("Backtest Start Date")
    end_date = get_date_input("Backtest End Date")
    
    test_df = filter_by_date(df, start_date, end_date)
    print(f"Backtesting on {len(test_df)} records.")
    
    if len(test_df) == 0:
        print("No data in range.")
        return
    
    FEATURES_COLUMNS = ['rsi', 'macd', 'macd_signal', 'macd_diff', 
                        'bb_high', 'bb_low', 'bb_mavg', 'sma_20', 'ema_12', 'adx']
                        
    strategy = TradingStrategy(model, buy_threshold=0.55, sell_threshold=0.45)
    results_df = strategy.backtest(test_df, FEATURES_COLUMNS)
    
    backtester = Backtester(initial_balance=10000)
    final_df = backtester.run(results_df)
    
    backtester.print_performance_metrics(final_df)
    backtester.plot_results(final_df)

def real_time_simulation_workflow(model, symbol):
    if model is None:
        print("No model loaded. Please train or load a model first.")
        return

    print("\n--- Real-Time Simulation ---")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {config.TIMEFRAME}")
    print("Press Ctrl+C to stop.")
    
    sim = RealTimeSimulator(model, symbol, config)
    sim.start()

def start_real_trading_workflow(model, symbol):
    if model is None:
        print("No model loaded. Please train or load a model first.")
        return

    print("\n⚠️ --- REAL TRADING MODE --- ⚠️")
    print("WARNING: This mode will execute REAL trades on Binance using your API keys.")
    confirm = input("Are you sure you want to proceed? (yes/no): ").lower()
    if confirm != 'yes':
        print("Aborted.")
        return

    mode = input("Select Mode (SPOT/FUTURES) [Default: SPOT]: ").strip().upper() or 'SPOT'
    
    # API Key Handling
    if config.BINANCE_API_KEY == "YOUR_BINANCE_API_KEY":
        print("API Keys not configured in config.py.")
        api_key = input("Enter Binance API Key: ").strip()
        api_secret = input("Enter Binance API Secret: ").strip()
        
        # Temporary config override
        class TempConfig:
            def __getattr__(self, name):
                return getattr(config, name)
        
        temp_config = TempConfig()
        temp_config.BINANCE_API_KEY = api_key
        temp_config.BINANCE_API_SECRET = api_secret
        used_config = temp_config
    else:
        used_config = config

    try:
        bot = RealTradingBot(model, symbol, used_config, mode=mode)
        bot.start()
    except Exception as e:
        print(f"Failed to start Real Trading Bot: {e}")

def main():
    print("Initializing...")
    global CURRENT_SYMBOL
    
    # Initial symbol selection
    sym = input(f"Enter symbol [Default: {CURRENT_SYMBOL}]: ").strip().upper()
    if sym:
        CURRENT_SYMBOL = sym
        
    df = get_data(CURRENT_SYMBOL)
    
    current_model = None
    
    while True:
        print(f"\n=== AnyCrypto Trading Bot ({CURRENT_SYMBOL}) ===")
        print(f"Current Model: {current_model}")
        print("1. Train New Model")
        print("2. Load Existing Model")
        print("3. Backtest Model")
        print("4. Change Symbol / Update Data")
        print("5. Start Real-Time Simulation")
        print("6. Start REAL Trading (Binance)")
        print("7. Exit")
        
        choice = input("Select option: ")
        
        if choice == '1':
            if df is not None:
                model = train_workflow(df)
                if model:
                    current_model = model
            else:
                print("No data available.")
        elif choice == '2':
            model = load_workflow()
            if model:
                current_model = model
        elif choice == '3':
            if df is not None:
                backtest_workflow(df, current_model)
            else:
                print("No data available.")
        elif choice == '4':
            new_df = change_symbol()
            if new_df is not None:
                df = new_df
        elif choice == '5':
            if current_model:
                real_time_simulation_workflow(current_model, CURRENT_SYMBOL)
            else:
                print("Please load or train a model first.")
        elif choice == '6':
            if current_model:
                start_real_trading_workflow(current_model, CURRENT_SYMBOL)
            else:
                print("Please load or train a model first.")
        elif choice == '7':
            print("Exiting.")
            break
        else:
            print("Invalid option.")

if __name__ == "__main__":
    main()
