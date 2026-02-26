import ccxt
import pandas as pd
import time
import logging
import datetime
import sys
import os

# Ensure src is in path for imports if running directly
sys.path.append(os.path.dirname(__file__))

from data_loader import fetch_data
from features import add_technical_indicators, preprocess_data
from strategy import TradingStrategy
import recovery # New Module
import csv

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealTradingBot:
    def __init__(self, model, symbol, config, mode='SPOT'):
        """
        Initialize Real Trading Bot.
        :param model: Trained sklearn model.
        :param symbol: Trading pair (e.g. 'BTC/USDT').
        :param config: Config object with API keys.
        :param mode: 'SPOT' or 'FUTURES'.
        """
        self.model = model
        self.symbol = symbol
        self.config = config
        self.mode = mode.upper()
        self.running = False
        self.position = None # Track current position state locally
        
        # Post-Stop-Loss Cooldown State
        self.stop_loss_cooldown = 0      # ciclos restantes de pausa tras un SL
        self.consecutive_buy_signals = 0  # señales BUY consecutivas acumuladas
        self.last_trade_reason = ''       # razón del último cierre
        
        # Initialize Exchange
        exchange_class = ccxt.binance
        self.exchange = exchange_class({
            'apiKey': config.BINANCE_API_KEY,
            'secret': config.BINANCE_API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future' if self.mode == 'FUTURES' else 'spot'
            }
        })
        
        # Load Markets
        try:
            self.exchange.load_markets()
            logger.info("Markets loaded.")
        except Exception as e:
            logger.error(f"Failed to load markets: {e}")
            raise

        # Strategy
        self.strategy = TradingStrategy(
            model, 
            buy_threshold=getattr(config, 'BUY_THRESHOLD', 0.6), 
            sell_threshold=getattr(config, 'SELL_THRESHOLD', 0.4)
        )
        
        self.csv_file = f"real_trading_{symbol.replace('/', '_')}_{self.mode.lower()}.csv"
        self._initialize_csv()

    def _initialize_csv(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Action', 'Price', 'PnL', 'Quantity', 'Balance', 'Info'])

    def log_trade(self, action, price, pnl=0, quantity=0, balance=0, info=""):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"{action} @ {price} | {info}")
        with open(self.csv_file, 'a', newline='') as f:
            csv.writer(f).writerow([timestamp, action, price, pnl, quantity, balance, info])


    def fetch_balance(self):
        """Fetch and display account balance."""
        try:
            balance = self.exchange.fetch_balance()
            if self.mode == 'FUTURES':
                usdt = balance['total']['USDT']
                free = balance['free']['USDT']
                print(f"📊 Futures Balance: Total: ${usdt:.2f} | Free: ${free:.2f}")
                return free
            else:
                # For Spot, check base and quote
                base_asset = self.symbol.split('/')[0]
                quote_asset = self.symbol.split('/')[1]
                
                base_bal = balance.get(base_asset, {}).get('free', 0)
                quote_bal = balance.get(quote_asset, {}).get('free', 0)
                
                print(f"📊 Spot Balance: {base_asset}: {base_bal} | {quote_asset}: ${quote_bal:.2f}")
                return quote_bal
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return 0

    def start(self):
        """Start the real trading loop."""
        self.running = True
        logger.info(f"🚀 Starting Real Trading Bot ({self.mode}) for {self.symbol}...")
        self.fetch_balance()
        
        # Recovery & Sync
        logger.info("Checking for recovery...")
        recovered_pos, last_bal = recovery.get_last_state(self.csv_file)
        if recovered_pos:
            logger.info(f"Trace found in logs: {recovered_pos}")
            self.position = recovered_pos
            
        # Always sync with exchange on startup in Real Trading
        synced_pos, msg = recovery.sync_with_exchange(self.exchange, self.symbol, self.position, self.mode)
        logger.info(f"Exchange Sync Result: {msg}")
        self.position = synced_pos
        
        try:
            while self.running:
                self._run_cycle()
                
                interval = getattr(self.config, 'CHECK_INTERVAL_SECONDS', 60)
                logger.info(f"Sleeping {interval}s...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")
        except Exception as e:
            logger.error(f"Bot error: {e}")
        finally:
            self.stop()

    def stop(self):
        self.running = False
        logger.info("Bot stopped.")

    def _run_cycle(self):
        """Execute one trading cycle."""
        # 1. Fetch Data
        df = self._get_latest_data()
        if df is None: return

        last_row = df.iloc[-1]
        current_price = last_row['close']
        
        # 2. Get Signal
        features_row = last_row
        if hasattr(self.model, 'feature_names_in_'):
            try:
                features_row = last_row[self.model.feature_names_in_]
            except KeyError:
                logger.error("Missing features.")
                return

        signal = self.strategy.get_signal(features_row, context_row=last_row)
        logger.info(f"Price: ${current_price:.2f} | Signal: {signal}")

        # 3. Execute
        self._manage_trade(signal, current_price)

    def _get_latest_data(self):
        """Fetch updated OHLCV data."""
        try:
            # Re-using fetch_data from data_loader which uses ccxt or requests
            # Or use self.exchange directly for consistency
            timeframe = getattr(self.config, 'TIMEFRAME', '1h')
            limit = 100 # Enough for indicators
            
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            df = add_technical_indicators(df)
            df = preprocess_data(df)
            return df
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None

    def _manage_trade(self, signal, price):
        """Place orders based on signals."""
        risk_per_trade = getattr(self.config, 'RISK_PER_TRADE', 0.02)
        balance = self.fetch_balance()
        amount_to_invest = balance * 0.95
        
        if amount_to_invest < 10:
            logger.warning("Insufficient balance to trade (Min $10).")
            return

        quantity = amount_to_invest / price
        
        # --- COOLDOWN POST-STOP LOSS ---
        if self.stop_loss_cooldown > 0:
            self.stop_loss_cooldown -= 1
            self.consecutive_buy_signals = 0
            logger.info(
                f"⏸️  POST-SL COOLDOWN: {self.stop_loss_cooldown} ciclos restantes. "
                f"Señal actual: {signal}. No se operará."
            )
            return

        required_confirmations = getattr(self.config, 'REENTRY_SIGNAL_CONFIRMATION', 2)
        
        try:
            if signal == 1:  # Señal alcista
                if self.mode == 'SPOT':
                    base_bal = self.exchange.fetch_balance()[self.symbol.split('/')[0]]['free']
                    if base_bal * price < 10:  # Efectivamente sin posición
                        self.consecutive_buy_signals += 1
                        if self.consecutive_buy_signals >= required_confirmations:
                            logger.info(
                                f"✅ Señal BUY confirmada ({self.consecutive_buy_signals}/{required_confirmations}). "
                                f"Ejecutando orden."
                            )
                            # UNCOMMENT TO ENABLE REAL TRADING
                            # self.exchange.create_market_buy_order(self.symbol, quantity)
                            print(f"⚠️ REAL TRADING MODE: WOULD BUY {quantity:.4f} {self.symbol} NOW.")
                            self.position = {'entry_price': price, 'quantity': quantity, 'type': 'LONG'}
                            self.last_trade_reason = ''
                            self.consecutive_buy_signals = 0
                            self.log_trade("BUY", price, 0, quantity, balance, "Signal Confirmed")
                        else:
                            logger.info(
                                f"🔍 Señal BUY ({self.consecutive_buy_signals}/{required_confirmations}). "
                                f"Esperando confirmación."
                            )
                    else:
                        self.consecutive_buy_signals = 0  # Ya tenemos posición
                        
            elif signal == -1:  # Señal bajista
                self.consecutive_buy_signals = 0
                if self.mode == 'SPOT':
                    base_bal = self.exchange.fetch_balance()[self.symbol.split('/')[0]]['free']
                    if base_bal * price > 10:  # Tenemos posición abierta
                        logger.info(f"Ejecutando SELL {self.symbol}...")
                        # UNCOMMENT TO ENABLE REAL TRADING
                        # self.exchange.create_market_sell_order(self.symbol, base_bal)
                        print(f"⚠️ REAL TRADING MODE: WOULD SELL {base_bal:.4f} {self.symbol} NOW.")
                        
                        pnl = 0
                        if self.position and 'entry_price' in self.position:
                            entry = self.position['entry_price']
                            pnl = (price - entry) * base_bal
                        
                        # Detectar si fue Stop Loss (precio cayó más del SL configurado)
                        sl_pct = getattr(self.config, 'STOP_LOSS_PCT', 0.02)
                        if self.position and price < self.position['entry_price'] * (1 - sl_pct):
                            self.last_trade_reason = 'STOP_LOSS'
                            cooldown = getattr(self.config, 'COOLDOWN_CANDLES_AFTER_SL', 3)
                            self.stop_loss_cooldown = cooldown
                            logger.warning(
                                f"⚠️  STOP LOSS ejecutado. Cooldown de {cooldown} ciclos activado."
                            )
                        
                        self.log_trade("SELL", price, pnl, base_bal, balance, "Signal Triggered")
                        self.position = None
            else:  # Hold
                self.consecutive_buy_signals = 0

        except Exception as e:
            logger.error(f"Order Execution Failed: {e}")
