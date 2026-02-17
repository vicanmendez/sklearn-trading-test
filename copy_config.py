# Configuration for Trading Bot

# Binance API Keys (Required for fetching data and trading operations)
BINANCE_API_KEY = "YOUR KEY HERE"
BINANCE_API_SECRET = "YOUR BINANCE API SECRET HERE"

# Email Configuration (Brevo / SMTP)
# Required for sending email alerts
EMAIL_ENABLED = True
BREVO_API_KEY = "BREVO KEY HERE"  # SMTP Password
BREVO_SENDER_EMAIL = "YOUR EMAIL HERE" # e.g. trading@yourdomain.com
EMAIL_RECIPIENT = "YOUR EMAIL HERE" # e.g. youremail@gmail.com
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
TAKE_PROFIT_PCT = 0.03     # 3% Take Profit (original 4%)
