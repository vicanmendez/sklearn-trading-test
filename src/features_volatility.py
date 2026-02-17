import pandas as pd
import numpy as np
import ta

def add_volatility_indicators(df):
    """
    Add advanced volatility indicators to the dataframe.
    """
    df = df.copy()
    
    # Standard Indicators (Reuse or re-implement for independence)
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
    df['bb_width'] = bollinger.bollinger_wband() # Width = (High - Low) / Avg
    df['bb_pband'] = bollinger.bollinger_pband() # %B
    
    # ATR (Absolute and Percentage)
    atr_ind = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
    df['atr'] = atr_ind.average_true_range()
    df['atr_pct'] = df['atr'] / df['close']
    
    # ADX (Trend Strength)
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
    
    # Rate of Change (Momentum)
    df['roc'] = ta.momentum.ROCIndicator(df['close'], window=12).roc()
    
    # SMA / EMA (Required for features)
    df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
    df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
    
    return df

def create_target_barrier(df, profit_pct=0.015, stop_pct=0.01, lookahead=24):
    """
    Create a Multi-Class Target.
    0: Neutral (Stop Hit or Timeout)
    1: Long Win (Price hits +Profit before -Stop)
    2: Short Win (Price hits -Profit before +Stop)
    """
    df = df.copy()
    targets = []
    
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)
    
    for i in range(n):
        if i + 1 >= n:
            targets.append(0)
            continue
            
        entry_price = closes[i]
        
        # Long Parameters
        long_tp = entry_price * (1 + profit_pct)
        long_sl = entry_price * (1 - stop_pct)
        
        # Short Parameters
        short_tp = entry_price * (1 - profit_pct)
        short_sl = entry_price * (1 + stop_pct)
        
        outcome = 0
        
        # Look forward
        end_idx = min(i + 1 + lookahead, n)
        
        # Status flags
        long_alive = True
        short_alive = True
        
        for j in range(i + 1, end_idx):
            curr_low = lows[j]
            curr_high = highs[j]
            
            # Check Long
            if long_alive:
                if curr_low <= long_sl:
                    long_alive = False
                elif curr_high >= long_tp:
                    outcome = 1
                    break
            
            # Check Short
            if short_alive:
                if curr_high >= short_sl:
                    short_alive = False
                elif curr_low <= short_tp:
                    outcome = 2
                    break
            
            if not long_alive and not short_alive:
                break
        
        targets.append(outcome)
        
    df['target'] = targets
    return df

def preprocess_data(df):
    """
    Clean data.
    """
    df = df.dropna()
    return df
