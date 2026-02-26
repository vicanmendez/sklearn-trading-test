# Configuration for Trading Bot

# Binance API Keys (Required for fetching data and trading operations)
BINANCE_API_KEY = "TU API KEY DE BINANCE AQUÍ"
BINANCE_API_SECRET = "TU API SECRET DE BINANCE AQUÍ"

# Google Gemini API Key (for AI market analysis)
GEMINI_API_KEY = "TU API KEY DE GEMINI AQUÍ"  # Set this in the webapp Settings panel

# Email Configuration (Brevo / SMTP)
# Required for sending email alerts --OPTIONAL
EMAIL_ENABLED = True
BREVO_API_KEY = "TU API KEY DE BREVO AQUÍ"  # SMTP Password
BREVO_SENDER_EMAIL = "TU EMAIL AQUÍ" # e.g. trading@yourdomain.com
EMAIL_RECIPIENT = "TU EMAIL AQUÍ" # e.g. youremail@gmail.com
SMTP_SERVER = "smtp-relay.brevo.com"
SMTP_PORT = 587

# Simulation Settings
SIMULATION_CAPITAL = 1000  # Starting capital in USDT
LEVERAGE = 1               # Leverage for Futures simulation
TIMEFRAME = '1h'           # Timeframe to trade (1h, 15m, etc.)
CHECK_INTERVAL_SECONDS = 60 # How often to check for new candles (in seconds)

# Trading Parameters
BUY_THRESHOLD = 0.60       # Minimum probability to BUY (original 0.6)
SELL_THRESHOLD = 0.40      # Maximum probability to SELL (below this = sell, original=0.4)
RISK_PER_TRADE = 0.02      # 2% of capital per trade
STOP_LOSS_PCT = 0.02       # 2% Stop Loss (original 2%)
TAKE_PROFIT_PCT = 0.03     # 8% Take Profit (original 4%)

# Post-Stop-Loss Cooldown Settings
# After a stop loss, the bot waits this many candles before considering re-entry.
# With TIMEFRAME='1h', COOLDOWN_CANDLES_AFTER_SL=3 means 3 hours of pause.
COOLDOWN_CANDLES_AFTER_SL = 3

# Number of consecutive BUY signals required to re-enter after cooldown ends.
# Avoids entering on a single spike signal. E.g., 2 means 2 consecutive bullish candles.
REENTRY_SIGNAL_CONFIRMATION = 2
