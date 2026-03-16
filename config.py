import os

# Pull settings from environment variables for flexibility & sec
# Pull settings from environment variables for flexibility & security
PO_EMAIL = os.getenv('PO_EMAIL')
PO_PASSWORD = os.getenv('PO_PASSWORD')
DEMO_MODE = os.getenv('DEMO_MODE', 'True').lower() == 'true'

PO_API_WSS = os.getenv('PO_API_WSS')
RISK_PERCENTAGE = float(os.getenv('RISK_PERCENTAGE', '2'))
MAX_TRADES_PER_DAY = int(os.getenv('MAX_TRADES_PER_DAY', '3'))
MIN_SIGNAL_SCORE = int(os.getenv('MIN_SIGNAL_SCORE', '4'))
MAX_DAILY_DRAWDOWN_PCT = float(os.getenv('MAX_DAILY_DRAWDOWN_PCT', '5'))
TRADE_COOLDOWN_MINUTES = int(os.getenv('TRADE_COOLDOWN_MINUTES', '5'))
MARKET_DATA_LIMIT = int(os.getenv('MARKET_DATA_LIMIT', '240'))

# AI settings
ENABLE_AI_FILTER = os.getenv('ENABLE_AI_FILTER', 'True').lower() == 'true'
AI_MODEL_PATH = os.getenv('AI_MODEL_PATH', 'trading_brain.pkl')
TRADE_MEMORY_PATH = os.getenv('TRADE_MEMORY_PATH', 'trade_memory.csv')
TRADE_JOURNAL_PATH = os.getenv('TRADE_JOURNAL_PATH', 'trade_journal.csv')
AI_MIN_BUY_PROB = float(os.getenv('AI_MIN_BUY_PROB', '0.60'))
AI_MAX_SELL_PROB = float(os.getenv('AI_MAX_SELL_PROB', '0.40'))
AI_RETRAIN_EVERY_N_TRADES = int(os.getenv('AI_RETRAIN_EVERY_N_TRADES', '25'))
AI_MIN_RETRAIN_ROWS = int(os.getenv('AI_MIN_RETRAIN_ROWS', '40'))

PAIRS = ['EURUSD', 'GBPUSD']
TIMEFRAME = '2m'
START_BALANCE = 1000  # Default demo balance

# If you want to add more pairs, do it here!