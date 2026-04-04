import os

# Pull settings from environment variables for flexibility & sec
PO_EMAIL = os.getenv('PO_EMAIL')
PO_PASSWORD = os.getenv('PO_PASSWORD')
DEMO_MODE = os.getenv('DEMO_MODE', 'True').lower() == 'true'

PO_API_WSS = os.getenv('PO_API_WSS')
PO_BALANCE_API = os.getenv('PO_BALANCE_API', '')
PO_BASE_URL = os.getenv('PO_BASE_URL', '')
PO_API_TOKEN = os.getenv('PO_API_TOKEN', '')
PO_ACCOUNT_MODE = os.getenv('PO_ACCOUNT_MODE', 'demo')
PO_ORDER_DURATION_SEC = int(os.getenv('PO_ORDER_DURATION_SEC', '60'))
PO_POLL_INTERVAL_SEC = float(os.getenv('PO_POLL_INTERVAL_SEC', '1.5'))
RISK_PERCENTAGE = float(os.getenv('RISK_PERCENTAGE', '2'))
MAX_TRADES_PER_DAY = int(os.getenv('MAX_TRADES_PER_DAY', '30'))
MIN_SIGNAL_SCORE = int(os.getenv('MIN_SIGNAL_SCORE', '4'))
MAX_DAILY_DRAWDOWN_PCT = float(os.getenv('MAX_DAILY_DRAWDOWN_PCT', '5'))
TRADE_COOLDOWN_MINUTES = int(os.getenv('TRADE_COOLDOWN_MINUTES', '5'))
MARKET_DATA_LIMIT = int(os.getenv('MARKET_DATA_LIMIT', '240'))
MARKET_DATA_EXCHANGES = [x.strip() for x in os.getenv('MARKET_DATA_EXCHANGES','kraken,coinbase').split(',') if x.strip()]

PAIRS = ['EURUSD', 'GBPUSD']
TIMEFRAME = '2m'
START_BALANCE = 1000  # Default demo balance
# Spec-driven controls
MAX_ACCOUNT_USE_PER_TRADE_PCT = float(os.getenv('MAX_ACCOUNT_USE_PER_TRADE_PCT', '25'))
STOP_LOSS_STAKE_PCT = float(os.getenv('STOP_LOSS_STAKE_PCT', '40'))
ANALYSIS_TIMEOUT_MINUTES = int(os.getenv('ANALYSIS_TIMEOUT_MINUTES', '10'))
TRAINING_PHASE = os.getenv('TRAINING_PHASE', 'sniper').lower()  # month1, month2, sniper

# AI settings
ENABLE_AI_FILTER = os.getenv('ENABLE_AI_FILTER', 'True').lower() == 'true'
AI_MODEL_PATH = os.getenv('AI_MODEL_PATH', 'trading_brain.pkl')
TRADE_MEMORY_PATH = os.getenv('TRADE_MEMORY_PATH', 'trade_memory.csv')
TRADE_JOURNAL_PATH = os.getenv('TRADE_JOURNAL_PATH', 'trade_journal.csv')
AI_MIN_BUY_PROB = float(os.getenv('AI_MIN_BUY_PROB', '0.60'))
AI_MAX_SELL_PROB = float(os.getenv('AI_MAX_SELL_PROB', '0.40'))
AI_RETRAIN_EVERY_N_TRADES = int(os.getenv('AI_RETRAIN_EVERY_N_TRADES', '1'))
AI_MIN_RETRAIN_ROWS = int(os.getenv('AI_MIN_RETRAIN_ROWS', '40'))

# If you want to add more pairs, do it here!
# Strategy ingestion artifacts
STRATEGY_KNOWLEDGE_PATH = os.getenv('STRATEGY_KNOWLEDGE_PATH', 'strategy_knowledge.json')
STRATEGY_REVIEW_STATE_PATH = os.getenv('STRATEGY_REVIEW_STATE_PATH', 'strategy_review_state.json')

PAIRS = ['EURUSD', 'GBPUSD']
TIMEFRAME = '1m'
HIGHER_TIMEFRAMES = ['1w', '1d', '4h', '1h']
START_BALANCE = 1000