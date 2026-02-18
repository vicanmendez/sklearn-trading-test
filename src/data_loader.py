import ccxt
import pandas as pd
import os
import time
from datetime import datetime

import yfinance as yf

def fetch_stock_data(symbol, start_date='2020-01-01', interval='1h'):
    """
    Fetch historical stock data using yfinance.
    """
    print(f"Fetching stock data for {symbol} from yfinance...")
    try:
        # yfinance interval mapping: 1h, 1d, etc.
        # yfinance max period for 1h is 730 days.
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, interval=interval)
        
        if df.empty:
            print(f"No data found for {symbol}")
            return None
            
        # Standardize columns
        df.reset_index(inplace=True)
        df = df.rename(columns={
            'Date': 'timestamp', 
            'Datetime': 'timestamp',
            'Open': 'open', 
            'High': 'high', 
            'Low': 'low', 
            'Close': 'close', 
            'Volume': 'volume'
        })
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Remove timezone if present to match crypto data usually
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            df.set_index('timestamp', inplace=True)
            
        # Keep only required columns
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        print(f"Fetched {len(df)} rows for {symbol}")
        return df
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None

def fetch_data(symbol='BTC/USDT', timeframe='1h', start_date='2020-01-01', interval=None):
    """
    Fetch historical data. Dispatches to crypto or stock based on symbol format.
    """
    if interval:
        timeframe = interval
        
    # Check if it's a crypto pair (contains '/')
    if '/' in symbol:
        return fetch_crypto_data(symbol, timeframe, start_date)
    else:
        # Try to normalize crypto symbol (e.g. XPLUSDT -> XPL/USDT)
        normalized_symbol = normalize_symbol(symbol)
        if normalized_symbol:
            print(f"Normalized {symbol} to {normalized_symbol}")
            return fetch_crypto_data(normalized_symbol, timeframe, start_date)
            
        return fetch_stock_data(symbol, start_date, timeframe)

def normalize_symbol(symbol):
    """
    Attempt to normalize a symbol string to a crypto pair format (BASE/QUOTE).
    """
    common_quotes = ['USDT', 'USDC', 'BUSD', 'DAI', 'BTC', 'ETH', 'BNB']
    
    for quote in common_quotes:
        if symbol.endswith(quote) and len(symbol) > len(quote):
            base = symbol[:-len(quote)]
            return f"{base}/{quote}"
            
    return None

def fetch_crypto_data(symbol='BTC/USDT', timeframe='1h', start_date='2020-01-01'):
    """
    Fetch historical OHLCV data from Binance using CCXT with pagination.
    """
    print(f"Fetching {symbol} data from Binance ({timeframe}) starting from {start_date}...")
    exchange = ccxt.binance()
    
    # ... (rest of original fetch_data logic) ...
    # Convert start_date to timestamp (ms)
    since = exchange.parse8601(f"{start_date}T00:00:00Z")
    
    all_ohlcv = []
    limit = 1000
    retries = 0
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            
            if not ohlcv:
                break
                
            all_ohlcv.extend(ohlcv)
            
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
            retries = 0 # Reset retries on success
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            retries += 1
            if retries > 3:
                print("Too many retries, stopping fetch.")
                break
            time.sleep(1) # Wait before retry
            
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
