import pandas as pd
import ta

def add_technical_indicators(df):
    """
    Add technical indicators to the dataframe.
    """
    df = df.copy()
    
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    df['bb_mavg'] = bollinger.bollinger_mavg()
    
    # SMA / EMA
    df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
    df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
    
    return df

def create_target(df):
    """
    Create target variable: 1 if next close > current close, else 0.
    """
    df = df.copy()
    # Target: 1 if the price rises in the next hour, 0 otherwise
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    return df

def preprocess_data(df):
    """
    Clean data (drop NaNs from indicators and last row target NaN).
    """
    df = df.dropna()
    return df

if __name__ == "__main__":
    # Test with dummy data or load if exists
    try:
        df = pd.read_csv('data/btc_1h.csv', index_col='timestamp', parse_dates=True)
        df = add_technical_indicators(df)
        df = create_target(df)
        df = preprocess_data(df)
        print(df.head())
        print(df.tail())
    except FileNotFoundError:
        print("Data file not found. Run data_loader.py first.")
