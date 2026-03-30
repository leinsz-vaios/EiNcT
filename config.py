import os

# --- AUTHENTICATION ---
PO_EMAIL = os.getenv('PO_EMAIL', 'your_email@example.com')
PO_PASSWORD = os.getenv('PO_PASSWORD', 'your_password')
PO_API_TOKEN = os.getenv('PO_API_TOKEN', '')
PO_BASE_URL = os.getenv('PO_BASE_URL', '')
PO_API_WSS = os.getenv('PO_API_WSS', 'wss://api-eu.po.market/socket.io/') # Added missing variable
PO_BALANCE_API = os.getenv('PO_BALANCE_API', '')

# --- MODE & ACCOUNT ---
DEMO_MODE = os.getenv('DEMO_MODE', 'True').lower() == 'true'
PO_ACCOUNT_MODE = os.getenv('PO_ACCOUNT_MODE', 'demo') 
START_BALANCE = 1000.0

# --- TRADING RULES ---
PAIRS = ['GBPJPY', 'EURJPY', 'USDCAD', 'EURUSD']
TIMEFRAME = '1m'
HIGHER_TIMEFRAMES = ['1w', '1d', '4h', '1h', '15m'] # Added missing variable

RISK_PERCENTAGE = 2.0
MAX_TRADES_PER_DAY = 500
MIN_SIGNAL_SCORE = 3
MAX_DAILY_DRAWDOWN_PCT = 10.0
TRADE_COOLDOWN_MINUTES = 1
MARKET_DATA_LIMIT = 240 # Added missing variable
MAX_ACCOUNT_USE_PER_TRADE_PCT = 10.0
STOP_LOSS_STAKE_PCT = 100.0
ANALYSIS_TIMEOUT_MINUTES = 10 # Added missing variable

# --- EXECUTION ---
PO_ORDER_DURATION_SEC = 60
PO_POLL_INTERVAL_SEC = 1.5

# --- AI & STRATEGY ---
ENABLE_AI_FILTER = True
AI_MODEL_PATH = 'trading_brain.pkl'
TRADE_MEMORY_PATH = 'trade_memory.csv'
TRADE_JOURNAL_PATH = 'trade_journal.csv'
STRATEGY_KNOWLEDGE_PATH = 'strategy_knowledge.json'
STRATEGY_REVIEW_STATE_PATH = 'strategy_review_state.json'
TRAINING_PHASE = 'month1'

AI_MIN_BUY_PROB = 0.55
AI_MAX_SELL_PROB = 0.45
AI_RETRAIN_EVERY_N_TRADES = 10
AI_MIN_RETRAIN_ROWS = 20