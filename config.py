import os

# Pull settings from environment variables for flexibility & sec
PO_EMAIL = os.getenv('PO_EMAIL')
PO_PASSWORD = os.getenv('PO_PASSWORD')
DEMO_MODE = os.getenv('DEMO_MODE', 'True').lower() == 'true'

PO_API_WSS = os.getenv('PO_API_WSS')
RISK_PERCENTAGE = float(os.getenv('RISK_PERCENTAGE', '2'))
MAX_TRADES_PER_DAY = int(os.getenv('MAX_TRADES_PER_DAY', '3'))

PAIRS = ['EURUSD', 'GBPUSD']
TIMEFRAME = '2m'
START_BALANCE = 1000  # Default demo balance

# If you want to add more pairs, do it here!