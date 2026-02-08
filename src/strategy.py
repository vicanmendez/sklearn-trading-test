class TradingStrategy:
    def __init__(self, model, buy_threshold=0.6, sell_threshold=0.4):
        self.model = model
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        
    def get_signal(self, features_row):
        """
        Get trading signal based on model prediction and thresholds.
        Returns: 1 (Buy), -1 (Sell), 0 (Hold)
        """
        # Reshape for prediction if single row
        X = features_row.values.reshape(1, -1)
        
        # Get probability of class 1 (Up)
        prob_up = self.model.predict_proba(X)[0][1]
        
        if prob_up > self.buy_threshold:
            return 1 # Buy
        elif prob_up < self.sell_threshold:
            return -1 # Sell
        else:
            return 0 # Hold

    def backtest(self, df, features_columns):
        """
        Simulate trading on historical data.
        """
        df = df.copy()
        signals = []
        probs = []
        
        # We need to iterate or Apply. Apply is faster but less flexible for stateful strategies.
        # Since this strategy is stateless (only depends on current row features), we can use apply/vectorized.
        
        X = df[features_columns]
        probs = self.model.predict_proba(X)[:, 1]
        
        df['prob_up'] = probs
        df['signal'] = 0
        
        df.loc[df['prob_up'] > self.buy_threshold, 'signal'] = 1
        df.loc[df['prob_up'] < self.sell_threshold, 'signal'] = -1
        
        return df
