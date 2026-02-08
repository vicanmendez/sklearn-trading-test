import ccxt
import pandas as pd
import os
import time
from datetime import datetime

def fetch_data(symbol='BTC/USDT', timeframe='1h', start_date='2020-01-01'):
    """
    Fetch historical OHLCV data from Binance using CCXT with pagination.
    """
    print(f"Fetching {symbol} data from Binance ({timeframe}) starting from {start_date}...")
    exchange = ccxt.binance()
    
    # Convert start_date to timestamp (ms)
    since = exchange.parse8601(f"{start_date}T00:00:00Z")
    
    all_ohlcv = []
    limit = 1000
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            
            if not ohlcv:
                break
                
            all_ohlcv.extend(ohlcv)
            
            # Update 'since' to the last timestamp + 1 timeframe duration
            # Actually, fetch_ohlcv returns candles starting >= since.
            # So next since should be the last timestamp + 1ms (to avoid duplicate if not careful, 
            # but usually just taking the last timestamp is safer if we handle duplicates later)
            # Better: use the timestamp of the last candle + 1ms
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 1
            
            # Print progress
            last_date = datetime.fromtimestamp(last_timestamp / 1000)
            print(f"Fetched up to {last_date} ({len(all_ohlcv)} candles)")
            
            # Stop if we reached current time (or close to it)
            if last_timestamp >= (time.time() * 1000) - 60*60*1000: # Within last hour
                break
                
            # Respect rate limits
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
    if not all_ohlcv:
        print("No data fetched.")
        return None
        
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Drop duplicates just in case
    df = df[~df.index.duplicated(keep='first')]
    
    print(f"Total fetched: {len(df)} rows.")
    return df

def get_filename(symbol, timeframe):
    # Convert symbol BTC/USDT -> BTC_USDT
    safe_symbol = symbol.replace('/', '_')
    return f"{safe_symbol}_{timeframe}.csv"

def save_data(df, symbol, timeframe='1h'):
    """
    Save dataframe to CSV.
    """
    if df is not None:
        if not os.path.exists('data'):
            os.makedirs('data')
        
        filename = get_filename(symbol, timeframe)
        filepath = os.path.join('data', filename)
        df.to_csv(filepath)
        print(f"Data saved to {filepath}")

def load_data(symbol, timeframe='1h'):
    """
    Load dataframe from CSV.
    """
    filename = get_filename(symbol, timeframe)
    filepath = os.path.join('data', filename)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        print(f"Data loaded from {filepath}")
        return df
    return None

if __name__ == "__main__":
    # Test
    df = fetch_data(symbol='ETH/USDT', start_date='2023-01-01')
    if df is not None:
        save_data(df, 'ETH/USDT')
        print(df.tail())
