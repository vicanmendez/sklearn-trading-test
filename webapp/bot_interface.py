import sys
import os
import pandas as pd
import threading
import time
import csv

# Add root and src to path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'src'))

from data_loader import fetch_data, save_data, load_data
from features import add_technical_indicators, create_target, preprocess_data
from models import train_models, get_best_model, save_model, load_model
from strategy import TradingStrategy
from simulator import RealTimeSimulator
from real_trading import RealTradingBot
import config

class BotInterface:
    def __init__(self):
        self.active_bots = {} # {bot_id: {'instance': bot_obj, 'thread': thread_obj, 'start_time': timestamp}}

    def get_data(self, symbol):
        # reuse get_data logic from main.py but adapted
        df = load_data(symbol)
        if df is None:
            try:
                # Fetch default logic
                pass 
            except Exception as e:
                print(f"Error fetching data: {e}")
                return None
        return df

    def fetch_and_update_data(self, symbol, start_date='2020-01-01'):
        """
        Explicitly fetch data from Binance (or source) and save it.
        Mirrors main.py get_data logic when data is missing or update requested.
        """
        try:
            print(f"Fetching data for {symbol} from {start_date}...")
            df = fetch_data(symbol, start_date=start_date)
            if df is not None and not df.empty:
                save_data(df, symbol)
                
                # Preprocess to ensure it's ready for training
                df = add_technical_indicators(df)
                df = create_target(df)
                df = preprocess_data(df)
                # save cleaned? usually fetch_data/save_data handles raw. 
                # main.py does preprocess on load. 
                # We can just save the raw data as fetch_data returns it (usually with some cleaning)
                # and let get_data handle the rest.
                
                return True, f"Data for {symbol} fetched and saved successfully. ({len(df)} records)"
            else:
                return False, "No data found or empty response."
        except Exception as e:
            print(f"Error in fetch_and_update_data: {e}")
            return False, str(e)

    def start_bot(self, bot_id, bot_config):
        """
        Start a bot instance.
        bot_config: dict containing 'pair', 'strategy', 'mode' (simulation/real), 'model_path'
        """
        if bot_id in self.active_bots:
            return False, "Bot already running"
            
        symbol = bot_config.get('pair', 'BTC/USDT')
        model_path = bot_config.get('model_path', 'models/best_model.pkl') # default
        mode = bot_config.get('mode', 'simulation')
        
        # Load Model
        if not os.path.exists(model_path):
             return False, f"Model not found at {model_path}"
             
        model = load_model(model_path)
        if model is None:
            return False, "Failed to load model"

        # Setup Config
        class DynamicConfig:
            def __getattr__(self, name):
                return getattr(config, name)
        
        d_config = DynamicConfig()
        
        try:
            if mode == 'simulation':
                bot_instance = RealTimeSimulator(model, symbol, d_config)
            elif mode == 'real':
                bot_instance = RealTradingBot(model, symbol, d_config, mode='SPOT') 
            else:
                return False, "Invalid mode"
                
            def run_wrapper():
                try:
                    bot_instance.start()
                except Exception as e:
                    print(f"Bot {bot_id} thread error: {e}")

            thread = threading.Thread(target=run_wrapper)
            thread.daemon = True
            thread.start()
            
            self.active_bots[bot_id] = {
                'instance': bot_instance,
                'thread': thread,
                'start_time': time.time(),
                'symbol': symbol,
                'mode': mode
            }
            print(f"DEBUG: Bot {bot_id} started. Active bots: {list(self.active_bots.keys())}")
            return True, "Bot started successfully"

        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"Failed to start bot: {e}"

    def stop_bot(self, bot_id):
        if bot_id in self.active_bots:
            bot_meta = self.active_bots[bot_id]
            instance = bot_meta['instance']
            if hasattr(instance, 'stop'):
                instance.stop()
            
            # Wait for thread? No, let it finish.
            del self.active_bots[bot_id]
            return True, "Bot stopped"
        return False, "Bot not running"

    def _format_runtime(self, seconds):
        if not seconds:
            return "00:00:00"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def get_bot_stats(self, bot_id):
        if bot_id not in self.active_bots:
            return None
            
        meta = self.active_bots[bot_id]
        instance = meta['instance']
        
        stats = {
            'status': 'running',
            'symbol': meta['symbol'],
            'mode': meta['mode'],
            'runtime': self._format_runtime(time.time() - meta['start_time']),
            'pnl': 0.0,
            'trades': 0
        }
        
        if hasattr(instance, 'capital') and hasattr(instance, 'initial_capital'):
             pnl = instance.capital - instance.initial_capital
             pnl_pct = (pnl / instance.initial_capital) * 100
             stats['pnl'] = round(pnl_pct, 2)
             stats['balance'] = instance.capital
        elif hasattr(instance, 'fetch_balance'): 
             pass
             
        if hasattr(instance, 'trades'):
            stats['trades'] = len(instance.trades)
            
        return stats

    def train_model(self, bot_id, symbol, start_date=None, end_date=None):
        try:
            # 1. Fetch Data
            # data_loader.fetch_data handles dynamic dispatch based on '/' in symbol
            df = fetch_data(symbol, start_date=start_date if start_date else '2020-01-01')
            if df is None or df.empty:
                return None, "No data found for symbol"
                
            # Filter by end_date if provided
            if end_date:
                df = df[df.index <= end_date]
                
            if len(df) < 100:
                return None, "Not enough data to train"
            
            # 2. Preprocess & Feature Engineering
            df = add_technical_indicators(df)
            df = create_target(df) # Creates 'target' column
            df = preprocess_data(df)
            
            # 3. Split Data
            # We use the last 20% for testing/validation
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]
            
            # Features (must match what's used in main.py/features.py)
            exclude_cols = ['target', 'open', 'high', 'low', 'close', 'volume']
            feature_cols = [c for c in df.columns if c not in exclude_cols]
            
            X_train = train_df[feature_cols]
            y_train = train_df['target']
            X_test = test_df[feature_cols]
            y_test = test_df['target']
            
            # 4. Train Model
            # Assuming train_models returns a dict of models or a single best model
            # For this integration, we'll assume a simplified flow or wrapper
            # If main.py has `train_models` returning dict, we get the best one
            
            from models import train_models, get_best_model
            trained_models = train_models(X_train, y_train)
            best_model = get_best_model(trained_models, X_test, y_test)
            
            if not best_model:
                return None, "Model training failed"

            # Save Model
            import time
            import os
            from models import save_model
            
            timestamp = int(time.time())
            filename = f"models/bot_{bot_id}_{symbol.replace('/', '')}_{timestamp}.pkl"
            full_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), filename)
            
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            save_model(best_model, full_path)
                
            # 5. Calculate Comparison Metrics (Model vs Buy & Hold) on TEST set
            # Buy & Hold Return
            initial_price = test_df.iloc[0]['close']
            final_price = test_df.iloc[-1]['close']
            bnh_return = ((final_price - initial_price) / initial_price) * 100
            
            # Model Return (Simplified Simulation)
            # Predict on test set
            predictions = best_model.predict(X_test)
            
            # Simulate: If pred=1 (Buy), hold for next candle. 
            # This is a rough approx matching the 'target' definition usually (next close > current close)
            # More accurate: Iterate and compound
            
            # Vectorized approx:
            # Shift close to get percent change for *next* period
            pct_change = test_df['close'].pct_change().shift(-1).fillna(0)
            
            # Strategy returns: if pred=1, we get pct_change. If pred=0, we get 0 (or risk free? no, just cash)
            # Align predictions with indices
            # predictions array matches X_test rows
            
            strategy_returns = pd.Series(predictions, index=test_df.index) * pct_change
            
            # Cumulative return
            # (1 + r1) * (1 + r2) ... - 1
            model_cum_return = (1 + strategy_returns).prod() - 1
            model_return_pct = model_cum_return * 100
            
            metrics = {
                "bnh_return": round(bnh_return, 2),
                "model_return": round(model_return_pct, 2),
                "samples": len(test_df)
            }
            
            msg = f"Training successful. Test Model Return: {metrics['model_return']}%, B&H: {metrics['bnh_return']}%"
            
            # Return True and the result dict including path and metrics
            return True, {
                'model_path': filename,
                'metrics': metrics,
                'message': msg
            }
            
        except Exception as e:
            print(f"Train error: {e}")
            return None, str(e)

    def run_backtest(self, bot_id, start_date=None, end_date=None):
        try:
            # Mocking backtest execution for UI demo (Parity with main.py workflow)
            # Real implementation would call strategy.backtest()
            
            # Simulating some processing time
            time.sleep(1)
            
            # Return dummy data structure compatible with the new Dashboard charts
            # In a real scenario, this would come from Backtester.run() -> final_df
            
            return True, {
                'total_return': '15.5%',
                'max_drawdown': '-5.2%',
                'sharpe_ratio': '1.8',
                'trades': 42,
                'equity_curve': [10000, 10100, 10050, 10200, 10350, 10300, 10500, 10800, 10700, 11000]
            }
        except Exception as e:
            return False, str(e)


    def get_bot_trades(self, bot_id):
        """
        Retrieve trade history for a specific bot from its CSV log.
        """
        if bot_id not in self.active_bots:
            # If not active, we might need to look up the bot in DB to get symbol/mode?
            # For now, only active bots. 
            # TODO: Support inactive bots by querying DB for config.
            return []
            
        meta = self.active_bots[bot_id]
        symbol = meta['symbol']
        mode = meta['mode']
        
        # Determine CSV filename based on mode
        if mode == 'simulation':
            filename = f"simulation_{symbol.replace('/', '_')}.csv"
        else:
            filename = f"real_trading_{symbol.replace('/', '_')}_{mode.lower()}.csv"
            
        trades = []
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        trades.append(row)
            except Exception as e:
                print(f"Error reading trades CSV: {e}")
                
        # Return reversed (newest first)
        return trades[::-1]

bot_interface = BotInterface()
