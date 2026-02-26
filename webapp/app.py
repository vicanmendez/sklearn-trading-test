from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from extensions import db, login_manager
from flask_login import login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import requests
import os
import importlib
import logging
import sys
import threading
import collections
import json

# --- Logging Setup ---
LOG_BUFFER = collections.deque(maxlen=500)

class ListHandler(logging.Handler):
    # Patterns to skip from the in-memory buffer (noisy HTTP access logs)
    _SKIP_PATTERNS = (
        '"GET /api/stats ',
        '"GET /api/logs ',
        '"GET /api/market ',
        'GET /static/',
        'LegacyAPIWarning',
    )
    def emit(self, record):
        try:
            msg = record.getMessage()
            if any(p in msg for p in self._SKIP_PATTERNS):
                return  # Skip noisy HTTP logs from UI buffer
            log_entry = self.format(record)
            LOG_BUFFER.append(log_entry)
        except Exception:
            self.handleError(record)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
list_handler = ListHandler()
list_handler.setFormatter(formatter)

# Configure Root Logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Avoid adding multiple handlers if reloaded
if not logger.handlers:
    logger.addHandler(list_handler)
    # Console handler for original stdout (terminal)
    console_handler = logging.StreamHandler(sys.__stdout__) 
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Redirect stdout/stderr to Logger with a Re-entrancy Guard
import threading as _threading

class StreamToLogger(object):
    _local = _threading.local()

    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        # Prevent infinite recursion if logging itself triggers a write to stdout/stderr
        if getattr(self._local, 'is_logging', False):
            # If we are already logging, write directly to the original stream to avoid loop
            if self.level == logging.ERROR:
                sys.__stderr__.write(buf)
            else:
                sys.__stdout__.write(buf)
            return

        self._local.is_logging = True
        try:
            for line in buf.rstrip().splitlines():
                clean_line = line.rstrip()
                if clean_line:
                    self.logger.log(self.level, clean_line)
        finally:
            self._local.is_logging = False

    def flush(self):
        pass

sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.ERROR)

# Initialize Flask app
app = Flask(__name__)

# Paths
webapp_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.dirname(webapp_dir)
instance_path = os.path.join(webapp_dir, 'instance')

if not os.path.exists(instance_path):
    os.makedirs(instance_path)

# App Config
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(instance_path, "database.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Register custom filter
@app.template_filter('from_json')
def from_json_filter(value):
    try:
        return json.loads(value)
    except:
        return {}

# Initialize extensions
db.init_app(app)
login_manager.init_app(app)
login_manager.login_view = 'login'

# Import models (must be after db initialization to avoid circular imports)
# Actually with extensions logic, we can import models at top IF models only imports db from extensions
# But let's keep it here or move to top if models.py doesn't import app
from db_models import User, Bot

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

from bot_interface import bot_interface
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from models import save_model

# Routes
@app.route('/')
def landing():
    # Fetch crypto data (mock or real)
    # Using CoinGecko API
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': 5,
            'page': 1,
            'sparkline': 'false'
        }
        response = requests.get(url, params=params, timeout=5)
        market_data = response.json()
    except:
        market_data = []

    # Get Bot Performance
    # We want to show all bots, but update stats for active ones
    bots = Bot.query.all()
    active_bots_data = []
    
    for bot in bots:
        bot_data = {
            'id': bot.id,
            'name': bot.name,
            'status': bot.status,
            'pnl': bot.pnl,
            'runtime': bot.runtime,
            'last_updated': bot.last_updated
        }
        
        # If running, get real-time stats
        if bot.status == 'running' or bot.status == 'training':
             stats = bot_interface.get_bot_stats(bot.id)
             if stats:
                 bot_data['runtime'] = stats.get('runtime', bot.runtime)
                 bot_data['pnl'] = stats.get('pnl', bot.pnl)
        
        active_bots_data.append(bot_data)

    return render_template('landing.html', active_bots=active_bots_data)

# -------------------------------------------------------
# Market Data API (crypto + stocks/ETFs, cached 60s)
# -------------------------------------------------------
_market_cache = {'data': None, 'ts': 0}
_MARKET_CACHE_TTL = 60  # seconds

STOCK_TICKERS = [
    {'symbol': 'AAPL',  'name': 'Apple',      'type': 'stock'},
    {'symbol': 'MSFT',  'name': 'Microsoft',  'type': 'stock'},
    {'symbol': 'GOOGL', 'name': 'Alphabet',   'type': 'stock'},
    {'symbol': 'AMZN',  'name': 'Amazon',     'type': 'stock'},
    {'symbol': 'NVDA',  'name': 'NVIDIA',     'type': 'stock'},
    {'symbol': 'SPY',   'name': 'S&P 500 ETF','type': 'etf'},
    {'symbol': 'QQQ',   'name': 'Nasdaq ETF', 'type': 'etf'},
    {'symbol': 'GLD',   'name': 'Gold ETF',   'type': 'etf'},
]

CRYPTO_IDS = ['bitcoin', 'ethereum', 'binancecoin', 'solana', 'ripple']

@app.route('/api/market')
def api_market():
    import time as _time
    now = _time.time()
    if _market_cache['data'] and (now - _market_cache['ts']) < _MARKET_CACHE_TTL:
        return jsonify({'status': 'success', 'data': _market_cache['data'], 'cached': True})

    result = {'crypto': [], 'stocks': [], 'etfs': []}

    # --- Crypto via CoinGecko ---
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            'vs_currency': 'usd',
            'ids': ','.join(CRYPTO_IDS),
            'order': 'market_cap_desc',
            'sparkline': 'false',
            'price_change_percentage': '24h'
        }
        resp = requests.get(url, params=params, timeout=8)
        for coin in resp.json():
            result['crypto'].append({
                'symbol': coin.get('symbol', '').upper(),
                'name': coin.get('name', ''),
                'price': coin.get('current_price', 0),
                'change_24h': round(coin.get('price_change_percentage_24h', 0) or 0, 2),
                'type': 'crypto'
            })
    except Exception as e:
        logger.warning(f"CoinGecko fetch failed: {e}")

    # --- Stocks & ETFs via yfinance ---
    try:
        import yfinance as yf
        symbols = [t['symbol'] for t in STOCK_TICKERS]
        tickers = yf.Tickers(' '.join(symbols))
        for meta in STOCK_TICKERS:
            sym = meta['symbol']
            try:
                info = tickers.tickers[sym].fast_info
                price = round(float(info.last_price), 2)
                prev  = round(float(info.previous_close), 2)
                change = round(((price - prev) / prev) * 100, 2) if prev else 0.0
                bucket = 'etfs' if meta['type'] == 'etf' else 'stocks'
                result[bucket].append({
                    'symbol': sym,
                    'name': meta['name'],
                    'price': price,
                    'change_24h': change,
                    'type': meta['type']
                })
            except Exception as inner_e:
                logger.warning(f"yfinance {sym} failed: {inner_e}")
    except Exception as e:
        logger.warning(f"yfinance fetch failed: {e}")

    _market_cache['data'] = result
    _market_cache['ts'] = now
    return jsonify({'status': 'success', 'data': result, 'cached': False})


# -------------------------------------------------------
# Market Insights API (Fear & Greed + AI Analysis, cached 1h)
# -------------------------------------------------------
_insights_cache = {'data': None, 'ts': 0}
_INSIGHTS_CACHE_TTL = 3600  # 1 hour

def _vix_to_fear_greed(vix: float) -> dict:
    """Convert VIX value to a 0-100 fear/greed score for stocks."""
    # VIX: <12 = extreme greed, 12-17 = greed, 17-25 = neutral,
    #       25-35 = fear, >35 = extreme fear
    if vix < 12:
        score, label = 90, 'Extreme Greed'
    elif vix < 17:
        score, label = 70, 'Greed'
    elif vix < 25:
        score, label = 50, 'Neutral'
    elif vix < 35:
        score, label = 30, 'Fear'
    else:
        score, label = 10, 'Extreme Fear'
    return {'value': score, 'label': label, 'vix': round(vix, 2)}

def _fetch_insights_data():
    """Fetch all insights data (called server-side, result cached 1h)."""
    import time as _time
    result = {
        'fear_greed': {
            'crypto': {'value': 50, 'label': 'Neutral'},
            'stocks': {'value': 50, 'label': 'Neutral', 'vix': None}
        },
        'ai_analysis': {'text': None, 'generated_at': None}
    }

    # --- Crypto Fear & Greed (alternative.me) ---
    try:
        resp = requests.get('https://api.alternative.me/fng/?limit=1', timeout=6)
        fng = resp.json()['data'][0]
        result['fear_greed']['crypto'] = {
            'value': int(fng['value']),
            'label': fng['value_classification']
        }
    except Exception as e:
        logger.warning(f"Crypto F&G fetch failed: {e}")

    # --- Stock Sentiment via VIX ---
    try:
        import yfinance as yf
        vix_data = yf.Ticker('^VIX').fast_info
        vix_val = float(vix_data.last_price)
        result['fear_greed']['stocks'] = _vix_to_fear_greed(vix_val)
    except Exception as e:
        logger.warning(f"VIX fetch failed: {e}")

    # --- Gemini AI Market Analysis ---
    try:
        import importlib, sys
        # Reload config to get latest key
        if 'config' in sys.modules:
            import config as _cfg
            importlib.reload(_cfg)
        else:
            import config as _cfg
        gemini_key = getattr(_cfg, 'GEMINI_API_KEY', '').strip()

        if gemini_key:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-3-flash-preview')

            # Build context from cached market data
            market = _market_cache.get('data') or {}
            crypto_prices = ', '.join(
                f"{c['symbol']} ${c['price']:,.2f} ({c['change_24h']:+.1f}%)"
                for c in market.get('crypto', [])
            ) or 'unavailable'
            stock_prices = ', '.join(
                f"{s['symbol']} ${s['price']:,.2f} ({s['change_24h']:+.1f}%)"
                for s in market.get('stocks', [])
            ) or 'unavailable'
            crypto_fg = result['fear_greed']['crypto']
            stock_fg  = result['fear_greed']['stocks']

            prompt = f"""You are a concise financial market analyst. Based on the following real-time data, 
write a brief (3-4 sentences) market panorama in English. Be objective and neutral. 
Do NOT give investment advice. Consider crypto, stocks and etfs.  make a relationship with the current global situation considering last news and the analyst opinions.

Current data:
- Crypto prices: {crypto_prices}
- Crypto Fear & Greed: {crypto_fg['value']}/100 ({crypto_fg['label']})
- Stock prices: {stock_prices}
- Stock market sentiment (VIX-based): {stock_fg['value']}/100 ({stock_fg['label']}) | VIX: {stock_fg.get('vix', 'N/A')}

Write only the analysis paragraph, no headers or bullet points."""

            response = model.generate_content(prompt)
            result['ai_analysis'] = {
                'text': response.text.strip(),
                'generated_at': _time.strftime('%Y-%m-%d %H:%M UTC', _time.gmtime())
            }
        else:
            result['ai_analysis'] = {
                'text': None,
                'generated_at': None,
                'error': 'No Gemini API key configured. Add it in Settings.'
            }
    except Exception as e:
        logger.warning(f"Gemini AI analysis failed: {e}")
        result['ai_analysis'] = {'text': None, 'generated_at': None, 'error': str(e)}

    return result

@app.route('/api/insights')
def api_insights():
    import time as _time
    now = _time.time()
    if _insights_cache['data'] and (now - _insights_cache['ts']) < _INSIGHTS_CACHE_TTL:
        return jsonify({'status': 'success', 'data': _insights_cache['data'], 'cached': True})

    # Run fetch in background thread to avoid blocking if called from landing page
    data = _fetch_insights_data()
    _insights_cache['data'] = data
    _insights_cache['ts'] = now
    return jsonify({'status': 'success', 'data': data, 'cached': False})


@app.route('/login', methods=['GET', 'POST'])

def login():
    # ... (login logic) ...
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
            
    return render_template('login.html')

# Configure logging
log_file = os.path.join(os.path.dirname(__file__), 'webapp.log')
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
# Ensure root logger has the file handler but avoid duplicates
if file_handler not in logging.getLogger().handlers:
    logging.getLogger().addHandler(file_handler)
logging.getLogger().setLevel(logging.INFO)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('landing'))



@app.route('/api/data/fetch', methods=['POST'])
@login_required
def fetch_data_api():
    data = request.get_json()
    symbol = data.get('symbol')
    start_date = data.get('start_date', '2020-01-01')
    
    if not symbol:
        return jsonify({'status': 'error', 'message': 'Symbol is required'}), 400
        
    try:
        # We need to use the logic from main.py's get_data/fetch_data
        # bot_interface.get_data calls load_data, but we want to force fetch if requested
        # We should add a method to bot_interface to handle explicit fetch
        
        # Using the imported helper from main.py context (via bot_interface imports or direct)
        # We imported save_model, let's allow bot_interface to handle this to keep app.py clean?
        # Or just call bot_interface.fetch_and_save_data(symbol, start_date)
        
        success, msg = bot_interface.fetch_and_update_data(symbol, start_date)
        if success:
            return jsonify({'status': 'success', 'message': msg})
        else:
            return jsonify({'status': 'error', 'message': msg}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/dashboard')
@login_required
def dashboard():
    bots = Bot.query.all()
    # Sync status again for dashboard
    for bot in bots:
        stats = bot_interface.get_bot_stats(bot.id)
        if stats:
            bot.status = 'running'
            bot.pnl = stats['pnl']
            bot.runtime = stats['runtime']
    db.session.commit()
    return render_template('dashboard.html', user=current_user, bots=bots)

@app.route('/profile', methods=['POST'])
@login_required
def update_profile():
    new_password = request.form.get('new_password')
    if new_password:
        current_user.password_hash = generate_password_hash(new_password)
        db.session.commit()
        flash('Password updated successfully')
    return redirect(url_for('dashboard'))

@app.route('/create_bot', methods=['POST'])
@login_required
def create_bot():
    name = request.form.get('name')
    asset_class = request.form.get('asset_class') # crypto or stock
    symbol = request.form.get('symbol')
    strategy = request.form.get('strategy')
    
    # Store initial config
    # Status is 'no_model' to enforce training/uploading
    config_data = json.dumps({'pair': symbol, 'strategy': strategy, 'asset_class': asset_class, 'mode': 'simulation'})
    new_bot = Bot(name=name, status='no_model', config=config_data)
    db.session.add(new_bot)
    db.session.commit()
    
    flash(f'Bot {name} created. Please Train or Upload a model to start.', 'info')
    return redirect(url_for('dashboard'))

@app.route('/api/stats')
@login_required
def api_stats():
    data = {}
    # Iterate over active bots in memory
    # Iterate over active bots in memory
    logger.debug(f"DEBUG: Active Bots Keys: {list(bot_interface.active_bots.keys())}")
    for bot_id in list(bot_interface.active_bots.keys()):
         stats = bot_interface.get_bot_stats(bot_id)
         logger.debug(f"DEBUG: Stats for {bot_id}: {stats}")
         if stats:
             data[bot_id] = stats
    return jsonify({'status': 'success', 'data': data})

@app.route('/api/bot/<int:bot_id>/trades')
@login_required
def bot_trades(bot_id):
    bot = Bot.query.get(bot_id)
    if not bot:
        return jsonify({'status': 'error', 'message': 'Bot not found'}), 404
        
    trades = bot_interface.get_bot_trades(bot_id)
    return jsonify({'status': 'success', 'data': trades})

# API for Bot Control
@app.route('/api/bot/<int:bot_id>/<action>', methods=['POST'])
@login_required
def bot_control(bot_id, action):
    # Log the request
    logger.info(f"Bot control requested: ID={bot_id}, Action={action}")
    
    bot = Bot.query.get(bot_id)
    if not bot:
        return jsonify({'status': 'error', 'message': f'Bot {bot_id} not found'}), 404
        
    data = request.get_json(silent=True) or {}
    
    try:
        config_dict = json.loads(bot.config) if bot.config else {}
    except:
        config_dict = {}

    if action == 'start':
        # Check if model exists
        if bot.status == 'no_model':
            # Double check if model path is set and valid
            model_path = config_dict.get('model_path')
            # Check relative to root_dir
            if not model_path or not os.path.exists(os.path.join(root_dir, model_path)):
                return jsonify({'status': 'error', 'message': 'Cannot start: No model found. Please Train or Upload a model first.'}), 400

        # ... (rest of start logic) ...
        # Ensure minimal config
        if 'pair' not in config_dict: config_dict['pair'] = 'BTC/USDT'
        
        # Prepare Config for Interface (Resolve Paths)
        # We pass a copy to avoid saving absolute paths to DB if we don't want to
        runtime_config = config_dict.copy()
        if 'model_path' in runtime_config:
            # Make absolute path for the interface
            # Fix: basedir -> root_dir
            runtime_config['model_path'] = os.path.join(root_dir, runtime_config['model_path'])
        
        success, msg = bot_interface.start_bot(bot.id, runtime_config)
        if success:
            bot.status = 'running'
            db.session.commit()
            return jsonify({'status': 'success', 'message': msg})
        else:
            return jsonify({'status': 'error', 'message': msg}), 400

    elif action == 'stop':
        success, msg = bot_interface.stop_bot(bot.id)
        if success:
            bot.status = 'stopped'
            db.session.commit()
            return jsonify({'status': 'success', 'message': msg})
        else:
             # Even if it failed (not running), mark as stopped
            bot.status = 'stopped'
            db.session.commit()
            return jsonify({'status': 'success', 'message': msg})
            
    elif action == 'delete':
        if bot.status in ['running', 'training']:
            return jsonify({'status': 'error', 'message': 'Cannot delete a running or training bot. Stop it first.'}), 400
            
        try:
            db.session.delete(bot)
            db.session.commit()
            return jsonify({'status': 'success', 'message': 'Bot deleted successfully'})
        except Exception as e:
            logger.exception(f"Error deleting bot {bot_id}: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 400

    elif action == 'train':
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        # Allow overriding symbol/ticker from request
        override_symbol = data.get('symbol') 
        
        symbol = override_symbol if override_symbol else config_dict.get('pair', 'BTC/USDT')

        def train_task(bot_id, sym, start, end):
            with app.app_context():
                bot = Bot.query.get(bot_id)
                # Parse config again inside thread
                try:
                    c_dict = json.loads(bot.config) if bot.config else {}
                except:
                    c_dict = {}
                
                bot.status = 'training'
                db.session.commit()
                logger.info(f"Started training for Bot {bot_id} on {sym}...")
                
                try:
                    success, result = bot_interface.train_model(bot_id, sym, start, end)
                    if success:
                        # result is a dict with model_path, metrics, message
                        model_path = result['model_path']
                        
                        c_dict['model_path'] = model_path
                        c_dict['metrics'] = result.get('metrics', {})
                        
                        if override_symbol:
                            c_dict['pair'] = override_symbol
                            
                        bot.config = json.dumps(c_dict)
                        bot.status = 'stopped' # Ready to start
                        msg = result.get('message', 'Training completed')
                        
                        logger.info(f"Bot {bot_id} Training Success: {msg}")
                    else:
                        bot.status = 'no_model'
                        msg = f"Training failed: {result}" # result is error msg string
                        logger.error(f"Bot {bot_id} Training Failed: {msg}")
                        
                except Exception as e:
                    bot.status = 'no_model'
                    msg = f"Training error: {e}"
                    logger.exception(f"Bot {bot_id} Training Exception: {e}")
                
                db.session.commit()
 
        import threading
        thread = threading.Thread(target=train_task, args=(bot.id, symbol, start_date, end_date))
        thread.start()
        
        return jsonify({'status': 'success', 'message': 'Training started. Monitor status/console for results.'})

    elif action == 'backtest':
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        try:
            config_dict = json.loads(bot.config) if bot.config else {}
        except:
            config_dict = {}
            
        success, result = bot_interface.run_backtest(bot.id, start_date, end_date, config_dict)
        if success:
            msg = f"Backtest Results: Return {result.get('total_return')}%"
            return jsonify({'status': 'success', 'message': msg, 'data': result})
        else:
            return jsonify({'status': 'error', 'message': result}), 400
            
    return jsonify({'status': 'error', 'message': 'Invalid action'}), 400

@app.route('/api/bot/<int:bot_id>/backtest', methods=['POST'])
@login_required
def bot_backtest(bot_id):
    data = request.get_json()
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    
    # We might need config to pass to backtest if needed, but bot_id should suffice if interface loads it
    # bot_interface.run_backtest signature is (bot_id, start, end) now
    
    success, result = bot_interface.run_backtest(bot_id, start_date, end_date)
    
    if success:
        return jsonify({'status': 'success', 'data': result})
    else:
        return jsonify({'status': 'error', 'message': result})

import config_manager
import config

@app.route('/api/config', methods=['GET', 'POST'])
@login_required
def api_config():
    if request.method == 'GET':
        keys = config_manager.get_config_values()
        # Mask secrets for display
        masked_keys = {}
        for k, v in keys.items():
            if 'SECRET' in k and len(v) > 8:
                masked_keys[k] = v[:4] + '*' * (len(v) - 8) + v[-4:]
            else:
                masked_keys[k] = v
        return jsonify({'status': 'success', 'config': masked_keys})
        
    elif request.method == 'POST':
        data = request.get_json()
        # Only allow specific keys
        allowed_keys = ['BINANCE_API_KEY', 'BINANCE_API_SECRET', 'BREVO_API_KEY']
        update_data = {k: data[k] for k in allowed_keys if k in data}
        
        success, msg = config_manager.update_config_values(update_data)
        if success:
            # Force reload config in app context if needed
            importlib.reload(config)
            return jsonify({'status': 'success', 'message': msg})
        else:
            return jsonify({'status': 'error', 'message': msg}), 500

from werkzeug.utils import secure_filename

# ... existing imports ...

@app.route('/api/bot/<int:bot_id>/upload_model', methods=['POST'])
@login_required
def upload_model(bot_id):
    bot = Bot.query.get_or_404(bot_id)
    
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400
        
    if file and file.filename.endswith('.pkl'):
        filename = secure_filename(f"uploaded_bot_{bot_id}_{file.filename}")
        models_dir = os.path.join(root_dir, 'models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        filepath = os.path.join(models_dir, filename)
        file.save(filepath)
        
        # Update config
        try:
            config_dict = json.loads(bot.config) if bot.config else {}
        except:
            config_dict = {}
            
        config_dict['model_path'] = f"models/{filename}"
        bot.config = json.dumps(config_dict)
        bot.status = 'stopped' # Ready to start
        db.session.commit()
        
        return jsonify({'status': 'success', 'message': f'Model uploaded and loaded: {filename}'})
        
    return jsonify({'status': 'error', 'message': 'Invalid file type. Only .pkl allowed'}), 400

@app.route('/api/logs')
@login_required
def api_logs():
    return jsonify({'status': 'success', 'logs': list(LOG_BUFFER)})


def create_initial_data():
    with app.app_context():
        db.create_all()
        # Create admin user if not exists
        if not User.query.filter_by(username='admin').first():
            admin = User(username='admin', password_hash=generate_password_hash('admin'))
            db.session.add(admin)
            db.session.commit()
            logger.info("Initial admin user created.")

if __name__ == '__main__':
    create_initial_data()
    # Sync zombie bot states: if server restarted, bots marked 'running'/'training' in DB
    # are no longer actually running (active_bots dict is empty). Mark them 'stopped'.
    with app.app_context():
        zombie_bots = Bot.query.filter(Bot.status.in_(['running', 'training'])).all()
        for zbot in zombie_bots:
            logger.info(f"[Startup] Bot {zbot.id} ({zbot.name}) was '{zbot.status}' — marking as 'stopped' (server restart)")
            zbot.status = 'stopped'
        if zombie_bots:
            db.session.commit()
    app.run(debug=True, port=5000)
