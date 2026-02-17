import numpy as np

class TradingStrategy:
    def __init__(self, model, buy_threshold=0.6, sell_threshold=0.4):
        self.model = model
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        
    def get_signal(self, features_row, context_row=None):
        """
        Get trading signal based on model prediction and thresholds, adjusted for market regime.
        Returns: 1 (Buy), -1 (Sell), 0 (Hold)
        
        context_row: Row containing 'close' and 'sma_200' if not in features_row.
        """
        # Reshape for prediction if single row
        # X = features_row.values.reshape(1, -1) # Old way causing warning
        
        # Create DataFrame with single row to preserve feature names
        import pandas as pd
        X = pd.DataFrame([features_row])
        
        # Get probability of class 1 (Up)
        prob_up = self.model.predict_proba(X)[0][1]
        
        # Determine Regime
        # If context_row is provided, use it. Otherwise assume features_row has the data if available.
        data_source = context_row if context_row is not None else features_row
        
        is_bullish = False
        if 'close' in data_source and 'sma_200' in data_source:
            if data_source['close'] > data_source['sma_200']:
                is_bullish = True
                
        # Adjust Thresholds based on Regime
        current_buy_threshold = self.buy_threshold
        current_sell_threshold = self.sell_threshold
        
        if is_bullish:
            # In Bull market, be more aggressive to buy, less aggressive to sell
            current_buy_threshold = max(0.5, self.buy_threshold - 0.1) # e.g. 0.6 -> 0.5
            current_sell_threshold = 0.3 # Lower sell threshold to avoid shaking out early, or keep as is?
            # actually, if we want to avoid selling in bull market, we should LOWER value required to trigger sell?
            # No, sell signal is when prob < sell_threshold. 
            # If we want to sell LESS, we need prob to be LOWER than a smaller number.
            current_sell_threshold = min(0.4, self.sell_threshold - 0.1)
            
        else:
            # Bear market - Defensive
            # Keep original thresholds or tighten them
            pass
            
        if prob_up > current_buy_threshold:
            return 1 # Buy
        elif prob_up < current_sell_threshold:
             # In bull market, maybe only sell if trend is broken? 
             # For now, just use probabilty.
            return -1 # Sell
        else:
            return 0 # Hold

    def backtest(self, df, features_columns):
        """
        Simulate trading on historical data.
        """
        df = df.copy()
        signals = []
        
        # We need to iterate or Apply. Apply is faster but less flexible for stateful strategies.
        # Since this strategy is stateless (only depends on current row features), we can use apply/vectorized.
        
        X = df[features_columns]
        probs = self.model.predict_proba(X)[:, 1]
        
        df['prob_up'] = probs
        df['signal'] = 0
        
        # Vectorized Regime Detection
        if 'sma_200' in df.columns:
            df['is_bullish'] = df['close'] > df['sma_200']
        else:
            df['is_bullish'] = False
            
        # Base thresholds
        buy_thr = self.buy_threshold
        sell_thr = self.sell_threshold
        
        # Adjusted thresholds
        # Bullish: Lower buy threshold (buy easier), Lower sell threshold (sell harder)
        df['adj_buy_thr'] = np.where(df['is_bullish'], max(0.5, buy_thr - 0.1), buy_thr)
        df['adj_sell_thr'] = np.where(df['is_bullish'], min(0.4, sell_thr - 0.1), sell_thr)
        
        df.loc[df['prob_up'] > df['adj_buy_thr'], 'signal'] = 1
        df.loc[df['prob_up'] < df['adj_sell_thr'], 'signal'] = -1
        
        return df
