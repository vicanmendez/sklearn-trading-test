import time
import datetime
import pandas as pd
import numpy as np
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import csv
import os

from data_loader import fetch_data
from features import add_technical_indicators, create_target, preprocess_data
from strategy import TradingStrategy
import recovery # New Module

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealTimeSimulator:
    def __init__(self, model, symbol, config):
        """
        Initialize the Real-Time Simulator.
        :param model: Trained sklearn model.
        :param symbol: Trading pair (e.g., 'BTC/USDT').
        :param config: Configuration module or object.
        """
        self.model = model
        self.symbol = symbol
        self.config = config
        self.running = False
        
        # Portfolio State
        self.capital = config.SIMULATION_CAPITAL
        self.initial_capital = config.SIMULATION_CAPITAL
        self.leverage = config.LEVERAGE if hasattr(config, 'LEVERAGE') else 1
        self.position = None  # None or dict with 'type', 'entry_price', 'quantity'
        self.trades = []
        
        # Strategy
        self.strategy = TradingStrategy(
            model, 
            buy_threshold=getattr(config, 'BUY_THRESHOLD', 0.6), 
            sell_threshold=getattr(config, 'SELL_THRESHOLD', 0.4)
        )
        
        self.csv_file = f"simulation_{symbol.replace('/', '_')}.csv"
        self._initialize_csv()
        
        # Recovery
        self._recover_state()

    def _recover_state(self):
        """Recover state from CSV."""
        logger.info("Checking for recovery...")
        recovered_pos, last_bal = recovery.get_last_state(self.csv_file)
        
        if recovered_pos:
            logger.info(f"Resuming Simulation Position: {recovered_pos}")
            self.position = recovered_pos
            # For Simulation, we need to know amount_invested to calculate PnL correctly later
            # recovery.py returns minimal info. We might need to estimate or assuming quantity * entry
            if 'amount_invested' not in self.position:
                 self.position['amount_invested'] = self.position['quantity'] * self.position['entry_price'] / self.leverage
                 
        if last_bal:
            logger.info(f"Restoring Capital: ${last_bal}")
            self.capital = float(last_bal)


    def start(self):
        """Start the simulation loop."""
        self.running = True
        logger.info(f"Starting Real-Time Simulation for {self.symbol}...")
        logger.info(f"Initial Capital: ${self.capital:.2f}, Leverage: {self.leverage}x")
        
        try:
            while self.running:
                self._run_cycle()
                
                # Sleep interval
                interval = getattr(self.config, 'CHECK_INTERVAL_SECONDS', 60)
                logger.info(f"Waiting {interval} seconds...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Simulation stopped by user.")
        except Exception as e:
            logger.error(f"Simulation error: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop the simulation."""
        self.running = False
        logger.info("Simulation stopped.")
        self._print_summary()

    def _run_cycle(self):
        """Execute one simulation cycle."""
        # 1. Fetch Data
        df = self._get_latest_data()
        if df is None or len(df) < 50:
            logger.warning("Not enough data to analyze.")
            return

        # 2. Analyze
        last_row = df.iloc[-1]
        current_price = last_row['close']
        current_time = last_row.name # timestamp index
        
        # Check Strategy Signal
        # context_row might be needed for regime detection (sma_200)
        
        # Filter features if model has feature_names_in_
        features_row = last_row
        if hasattr(self.model, 'feature_names_in_'):
            try:
                features_row = last_row[self.model.feature_names_in_]
            except KeyError as e:
                logger.error(f"Missing features in data: {e}")
                return

        signal = self.strategy.get_signal(features_row, context_row=last_row)
        prob = last_row.get('prob_up', 0.5) 
        
        logger.info(f"Price: ${current_price:.2f} | Signal: {signal}")

        # 3. Execute Trades
        self._manage_positions(signal, current_price, current_time)

    def _get_latest_data(self):
        """Fetch and preprocess latest data."""
        # Fetch slightly more history than needed for indicators (e.g., 200 SMA)
        # 1000 candles is safe.
        # Start date: calculate based on timeframe. 1000 hours ago approx 42 days.
        start_date = (datetime.datetime.now() - datetime.timedelta(days=60)).strftime('%Y-%m-%d')
        
        df = fetch_data(self.symbol, timeframe=getattr(self.config, 'TIMEFRAME', '1h'), start_date=start_date)
        if df is not None:
            df = add_technical_indicators(df)
            # Create target is for training, not needed for live prediction, but features needs to match training features.
            # Preprocess removes NaNs.
            df = preprocess_data(df)
        return df

    def _manage_positions(self, signal, price, timestamp):
        """Manage entry and exit of positions."""
        
        # Check Stop Loss / Take Profit if in position
        if self.position:
            self._check_exit_conditions(price, timestamp)
        
        # Entry Logic
        if not self.position:
            if signal == 1: # Buy Signal
                self._open_position('LONG', price, timestamp)
            elif signal == -1: # Sell Signal (for Shorting in Futures)
                # Only if leverage > 1 implying futures, or if we want to allow shorting.
                # Assuming Futures Simulator logic ported from babyshark allow Shorts.
                self._open_position('SHORT', price, timestamp)
                
        # Exit Logic based on Signal reversal
        elif self.position:
            if self.position['type'] == 'LONG' and signal == -1:
                self._close_position(price, timestamp, 'SIGNAL_REVERSAL')
            elif self.position['type'] == 'SHORT' and signal == 1:
                self._close_position(price, timestamp, 'SIGNAL_REVERSAL')

    def _open_position(self, type_, price, timestamp):
        """Open a new position."""
        # Calculate position size
        # Risk management: usually fixed amount or % of capital. 
        # Config usage: risk per trade or fixed capital.
        # Let's use entire capital * leverage for simplicity as per babyshark simulator?
        # Or a fraction.
        # Let's use 95% of available capital to account for fees.
        amount_invested = self.capital * 0.95
        quantity = (amount_invested * self.leverage) / price
        
        self.position = {
            'type': type_,
            'entry_price': price,
            'quantity': quantity,
            'amount_invested': amount_invested,
            'start_time': timestamp
        }
        
        
        log_msg = (
            f"\n{'='*40}\n"
            f"🚀 OPEN {type_} POSITION\n"
            f"Symbol: {self.symbol}\n"
            f"Price: ${price:.2f}\n"
            f"Quantity: {quantity:.4f}\n"
            f"Invested: ${amount_invested:.2f}\n"
            f"Portfolio Total: ${self.capital:.2f}\n"
            f"{'='*40}\n"
        )
        print(log_msg)
        logger.info(f"OPEN {type_} at ${price:.2f}")
        
        self._send_alert(f"OPEN {type_}", price, quantity, f"Signal Triggered | Portfolio: ${self.capital:.2f}")
        self._log_trade(timestamp, type_, price, quantity, amount_invested, "OPEN")

    def _close_position(self, price, timestamp, reason):
        """Close current position."""
        if not self.position:
            return

        pos = self.position
        entry_price = pos['entry_price']
        quantity = pos['quantity']
        leverage = self.leverage
        
        # Calculate PnL
        if pos['type'] == 'LONG':
            pnl_pct = (price - entry_price) / entry_price
        else: # SHORT
            pnl_pct = (entry_price - price) / entry_price
            
        pnl_pct *= leverage
        pnl_amount = pos['amount_invested'] * pnl_pct
        
        # Update Capital
        self.capital += pnl_amount
        # Note: In real trading, you get back margin + pnl.
        # Here capital tracks total equity.
        # If we invested amount_invested, we essentially "held" it. 
        # The logic: New Capital = Old Capital + PnL.
        
        log_msg = (
            f"\n{'='*40}\n"
            f"💰 CLOSE {pos['type']} POSITION\n"
            f"Symbol: {self.symbol}\n"
            f"Price: ${price:.2f}\n"
            f"Reason: {reason}\n"
            f"PnL: ${pnl_amount:.2f} ({pnl_pct*100:.2f}%)\n"
            f"Portfolio Total: ${self.capital:.2f}\n"
            f"{'='*40}\n"
        )
        print(log_msg)
        logger.info(f"CLOSE {pos['type']} at ${price:.2f} | PnL: ${pnl_amount:.2f}")

        self._send_alert(f"CLOSE {pos['type']}", price, quantity, f"{reason} | PnL: ${pnl_amount:.2f} | Portfolio: ${self.capital:.2f}")
        self._log_trade(timestamp, f"CLOSE_{pos['type']}", price, quantity, pnl_amount, reason)
        
        self.trades.append({
            'type': pos['type'],
            'entry': entry_price,
            'exit': price,
            'pnl': pnl_amount,
            'pnl_pct': pnl_pct,
            'reason': reason
        })
        
        self.position = None

    def _check_exit_conditions(self, price, timestamp):
        """Check Stop Loss and Take Profit."""
        if not self.position:
            return

        pos = self.position
        entry_price = pos['entry_price']
        
        if pos['type'] == 'LONG':
            pct_change = (price - entry_price) / entry_price
        else:
            pct_change = (entry_price - price) / entry_price
            
        # Leverage adjustments? SL/TP relative to entry price usually.
        # Config has SL/TP percentage (e.g. 0.02 for 2%).
        sl_pct = getattr(self.config, 'STOP_LOSS_PCT', 0.02)
        tp_pct = getattr(self.config, 'TAKE_PROFIT_PCT', 0.04)
        
        # Check
        if pct_change <= -sl_pct:
            self._close_position(price, timestamp, 'STOP_LOSS')
        elif pct_change >= tp_pct:
            self._close_position(price, timestamp, 'TAKE_PROFIT')

    def _send_alert(self, action, price, quantity, info):
        """Send email alert."""
        if not getattr(self.config, 'EMAIL_ENABLED', False):
            return

        try:
            sender = getattr(self.config, 'BREVO_SENDER_EMAIL', None)
            recipient = getattr(self.config, 'EMAIL_RECIPIENT', None)
            password = getattr(self.config, 'BREVO_API_KEY', None)
            server_host = getattr(self.config, 'SMTP_SERVER', 'smtp-relay.brevo.com')
            port = getattr(self.config, 'SMTP_PORT', 587)

            if not sender or not recipient or not password:
                logger.warning("Email configuration missing.")
                return

            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = recipient
            msg['Subject'] = f"🚨 TRADING ALERT: {action} {self.symbol}"

            body = f"""
            <h3>Trading Verification Alert</h3>
            <p><strong>Action:</strong> {action}</p>
            <p><strong>Symbol:</strong> {self.symbol}</p>
            <p><strong>Price:</strong> ${price:.2f}</p>
            <p><strong>Quantity:</strong> {quantity:.4f}</p>
            <p><strong>Info:</strong> {info}</p>
            <p><strong>Time:</strong> {datetime.datetime.now()}</p>
            <p><strong>Current Capital:</strong> ${self.capital:.2f}</p>
            """
            msg.attach(MIMEText(body, 'html'))

            with smtplib.SMTP(server_host, port) as server:
                server.starttls()
                server.login(sender, password)
                server.send_message(msg)
                
            logger.info(f"Email alert sent: {action}")

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    def _initialize_csv(self):
        """Initialize CSV log file."""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Action', 'Price', 'Quantity', 'Amount/PnL', 'Info', 'Total_Portfolio'])

    def _log_trade(self, timestamp, action, price, quantity, amount, info):
        """Log trade to CSV."""
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, action, price, quantity, amount, info, self.capital])

    def _print_summary(self):
        """Print simulation summary."""
        print("\n--- Simulation Summary ---")
        print(f"Final Capital: ${self.capital:.2f}")
        print(f"Total Trades: {len(self.trades)}")
        total_pnl = self.capital - self.initial_capital
        print(f"Net PnL: ${total_pnl:.2f} ({(total_pnl/self.initial_capital)*100:.2f}%)")

if __name__ == "__main__":
    # Test execution
    pass
