import os

# ====================
# ACCOUNT SETTINGS
# ====================
PO_EMAIL = os.getenv('PO_EMAIL')
PO_PASSWORD = os.getenv('PO_PASSWORD')
DEMO_MODE = os.getenv('DEMO_MODE', 'True').lower() == 'true'
START_BALANCE = 1000  # Default demo balance

# ====================
# POCKET OPTION API SETTINGS
# ====================
PO_API_WSS = os.getenv('PO_API_WSS')
PO_BALANCE_API = os.getenv('PO_BALANCE_API', '')
PO_BASE_URL = os.getenv('PO_BASE_URL', '')
PO_API_TOKEN = os.getenv('PO_API_TOKEN', '')
PO_ACCOUNT_MODE = os.getenv('PO_ACCOUNT_MODE', 'demo')
PO_ORDER_DURATION_SEC = int(os.getenv('PO_ORDER_DURATION_SEC', '60'))
PO_POLL_INTERVAL_SEC = float(os.getenv('PO_POLL_INTERVAL_SEC', '1.5'))

# ====================
# TRADING PAIRS & TIMEFRAMES
# ====================
PAIRS = ['EURUSD', 'GBPUSD', 'ETHUSD']
TIMEFRAME = '1m'
HIGHER_TIMEFRAMES = ['4h', '1h']

# ====================
# RISK MANAGEMENT
# ====================
RISK_PERCENTAGE = float(os.getenv('RISK_PERCENTAGE', '2'))
MAX_ACCOUNT_USE_PER_TRADE_PCT = float(os.getenv('MAX_ACCOUNT_USE_PER_TRADE_PCT', '5'))
STOP_LOSS_STAKE_PCT = float(os.getenv('STOP_LOSS_STAKE_PCT', '40'))
MAX_TRADES_PER_DAY = int(os.getenv('MAX_TRADES_PER_DAY', '20'))
MAX_DAILY_DRAWDOWN_PCT = float(os.getenv('MAX_DAILY_DRAWDOWN_PCT', '15'))
TRADE_COOLDOWN_MINUTES = int(os.getenv('TRADE_COOLDOWN_MINUTES', '5'))

# ====================
# ICT GATE SETTINGS (REQUIRED FOR HYBRID BOT!)
# ====================
ENABLE_TIME_FILTER = True  # Only trade during kill zones

# Silver Bullet: Time windows (in UTC)
LONDON_KILLZONE_UTC = (7, 10)   # 7am-10am UTC (London open)
NY_KILLZONE_UTC = (13, 16)      # 1pm-4pm UTC (NY open)

# Liquidity settings
LIQUIDITY_LOOKBACK = 20         # How many candles back to check for highs/lows
SWEEP_MEMORY_CANDLES = int(os.getenv('SWEEP_MEMORY_CANDLES', '5'))        # How many candles to remember a sweep

# ====================
# SCORING SYSTEM
# ====================
MIN_SIGNAL_SCORE = float(os.getenv('MIN_SIGNAL_SCORE', '4.0'))

# ====================
# AI SETTINGS
# ====================
ENABLE_AI_FILTER = os.getenv('ENABLE_AI_FILTER', 'True').lower() == 'true'
# Change these lines in config.py:
AI_MODEL_PATH = os.getenv('AI_MODEL_PATH', '/data/trading_brain.pkl')
TRADE_MEMORY_PATH = os.getenv('TRADE_MEMORY_PATH', '/data/trade_memory.csv')
TRADE_JOURNAL_PATH = os.getenv('TRADE_JOURNAL_PATH', '/data/trade_journal.csv')

AI_MIN_BUY_PROB = float(os.getenv('AI_MIN_BUY_PROB', '0.55'))
AI_MAX_SELL_PROB = float(os.getenv('AI_MAX_SELL_PROB', '0.45'))
AI_RETRAIN_EVERY_N_TRADES = int(os.getenv('AI_RETRAIN_EVERY_N_TRADES', '25'))
AI_MIN_RETRAIN_ROWS = int(os.getenv('AI_MIN_RETRAIN_ROWS', '40'))

# ====================
# TRAINING PHASES
# ====================
TRAINING_PHASE = os.getenv('TRAINING_PHASE', 'month1').lower()

# ====================
# MARKET DATA
# ====================
MARKET_DATA_LIMIT = int(os.getenv('MARKET_DATA_LIMIT', '240'))
MARKET_DATA_EXCHANGES = [x.strip() for x in os.getenv('MARKET_DATA_EXCHANGES', 'kraken,coinbase').split(',') if x.strip()]

# ====================
# STRATEGY KNOWLEDGE
# ====================
STRATEGY_KNOWLEDGE_PATH = os.getenv('STRATEGY_KNOWLEDGE_PATH', 'strategy_knowledge.json')
STRATEGY_REVIEW_STATE_PATH = os.getenv('STRATEGY_REVIEW_STATE_PATH', 'strategy_review_state.json')
ANALYSIS_TIMEOUT_MINUTES = int(os.getenv('ANALYSIS_TIMEOUT_MINUTES', '10'))