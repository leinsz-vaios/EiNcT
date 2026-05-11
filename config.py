import os

# ====================
# ACCOUNT SETTINGS
# ====================
DEMO_MODE     = os.getenv('DEMO_MODE', 'True').lower() == 'true'
START_BALANCE = float(os.getenv('START_BALANCE', '1000'))

# ====================
# KRAKEN API
# ====================
# Get these from:
# DEMO  → https://demo-futures.kraken.com  (free, no email verification)
# LIVE  → https://futures.kraken.com → Security → API
KRAKEN_API_KEY    = os.getenv('KRAKEN_API_KEY', '')
KRAKEN_API_SECRET = os.getenv('KRAKEN_API_SECRET', '')

# ====================
# TRADING PAIRS & TIMEFRAMES
# ====================
PAIRS            = ['EURUSD', 'GBPUSD', 'ETHUSD']
TIMEFRAME        = '1m'
HIGHER_TIMEFRAMES = ['4h', '1h']

# How long to hold each trade open (seconds)
TRADE_HOLD_SECONDS = int(os.getenv('TRADE_HOLD_SECONDS', '300'))  # 5 minutes default

# ====================
# RISK MANAGEMENT
# ====================
RISK_PERCENTAGE              = float(os.getenv('RISK_PERCENTAGE', '2'))
MAX_ACCOUNT_USE_PER_TRADE_PCT = float(os.getenv('MAX_ACCOUNT_USE_PER_TRADE_PCT', '5'))
STOP_LOSS_STAKE_PCT          = float(os.getenv('STOP_LOSS_STAKE_PCT', '40'))
MAX_TRADES_PER_DAY           = int(os.getenv('MAX_TRADES_PER_DAY', '20'))
MAX_DAILY_DRAWDOWN_PCT       = float(os.getenv('MAX_DAILY_DRAWDOWN_PCT', '15'))
TRADE_COOLDOWN_MINUTES       = int(os.getenv('TRADE_COOLDOWN_MINUTES', '5'))

# ====================
# ICT GATE SETTINGS
# ====================
ENABLE_TIME_FILTER   = True
LONDON_KILLZONE_UTC  = (7, 10)    # 7am-10am UTC
NY_KILLZONE_UTC      = (13, 16)   # 1pm-4pm UTC
LIQUIDITY_LOOKBACK   = 20
SWEEP_MEMORY_CANDLES = int(os.getenv('SWEEP_MEMORY_CANDLES', '5'))

# ====================
# SCORING SYSTEM
# ====================
MIN_SIGNAL_SCORE = float(os.getenv('MIN_SIGNAL_SCORE', '4.0'))

# ====================
# AI SETTINGS
# ====================
ENABLE_AI_FILTER          = os.getenv('ENABLE_AI_FILTER', 'True').lower() == 'true'
AI_MODEL_PATH             = os.getenv('AI_MODEL_PATH', '/data/trading_brain.pkl')
TRADE_MEMORY_PATH         = os.getenv('TRADE_MEMORY_PATH', '/data/trade_memory.csv')
TRADE_JOURNAL_PATH        = os.getenv('TRADE_JOURNAL_PATH', '/data/trade_journal.csv')
AI_MIN_BUY_PROB           = float(os.getenv('AI_MIN_BUY_PROB', '0.55'))
AI_MAX_SELL_PROB          = float(os.getenv('AI_MAX_SELL_PROB', '0.45'))
AI_RETRAIN_EVERY_N_TRADES = int(os.getenv('AI_RETRAIN_EVERY_N_TRADES', '25'))
AI_MIN_RETRAIN_ROWS       = int(os.getenv('AI_MIN_RETRAIN_ROWS', '40'))

# ====================
# TRAINING PHASES
# ====================
TRAINING_PHASE = os.getenv('TRAINING_PHASE', 'month1').lower()

# ====================
# MARKET DATA
# ====================
MARKET_DATA_LIMIT     = int(os.getenv('MARKET_DATA_LIMIT', '240'))
MARKET_DATA_EXCHANGES = ['kraken']  # spot data source

# ====================
# STRATEGY KNOWLEDGE
# ====================
STRATEGY_KNOWLEDGE_PATH   = os.getenv('STRATEGY_KNOWLEDGE_PATH', 'strategy_knowledge.json')
STRATEGY_REVIEW_STATE_PATH = os.getenv('STRATEGY_REVIEW_STATE_PATH', 'strategy_review_state.json')
ANALYSIS_TIMEOUT_MINUTES  = int(os.getenv('ANALYSIS_TIMEOUT_MINUTES', '10'))