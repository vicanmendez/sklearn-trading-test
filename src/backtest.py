import pandas as pd
import matplotlib.pyplot as plt

class Backtester:
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        
    def run(self, df):
        """
        Run backtest on dataframe with 'signal' column.
        Signal: 1 (Buy/Long), -1 (Sell/Short), 0 (Cash/Neutral)
        Assumption: Signal at row `t` executes at Close `t`.
        Returns at `t+1` depend on price change from `t` to `t+1`.
        """
        df = df.copy()
        
        # Calculate hourly returns of the asset
        df['hourly_return'] = df['close'].pct_change().shift(-1)
        
        # Strategy return (assuming we hold the position for the next hour)
        # If signal is 1, we get hourly_return.
        # If signal is -1, we get -hourly_return (short).
        # If signal is 0, we get 0.
        # Note: This is a simplified "vectorized" backtest.
        df['strategy_return'] = df['signal'] * df['hourly_return']
        
        # Buy & Hold return
        df['bnh_return'] = df['hourly_return']
        
        # Drop NaN (last row)
        df.dropna(subset=['strategy_return'], inplace=True)
        
        # Cumulative Returns
        df['cumulative_strategy_return'] = (1 + df['strategy_return']).cumprod()
        df['cumulative_bnh_return'] = (1 + df['bnh_return']).cumprod()
        
        # Equity Curve
        df['equity_strategy'] = self.initial_balance * df['cumulative_strategy_return']
        df['equity_bnh'] = self.initial_balance * df['cumulative_bnh_return']
        
        return df
        
    def plot_results(self, df):
        """
        Plot strategy vs Buy & Hold.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['equity_strategy'], label='ML Strategy')
        plt.plot(df.index, df['equity_bnh'], label='Buy & Hold', alpha=0.7)
        plt.title('Backtest Results: ML Strategy vs Buy & Hold')
        plt.xlabel('Date')
        plt.ylabel('Equiy ($)')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig('backtest_results.png')
        print("Plot saved to backtest_results.png")
        
    def print_performance_metrics(self, df):
        """
        Print metrics.
        """
        total_return_strategy = df['cumulative_strategy_return'].iloc[-1] - 1
        total_return_bnh = df['cumulative_bnh_return'].iloc[-1] - 1
        
        print(f"Total Return (Strategy): {total_return_strategy:.2%}")
        print(f"Total Return (Buy & Hold): {total_return_bnh:.2%}")
        
        # Win Rate
        wins = df[df['strategy_return'] > 0]
        losses = df[df['strategy_return'] < 0]
        win_rate = len(wins) / (len(wins) + len(losses)) if (len(wins) + len(losses)) > 0 else 0
        
        print(f"Win Rate: {win_rate:.2%}")
