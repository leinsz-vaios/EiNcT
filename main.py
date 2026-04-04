import warnings
# Ignore the specific DeprecationWarning about utcnow
warnings.filterwarnings(action="ignore", message="datetime.datetime.now(datetime.UTC)")
import argparse
import csv
import datetime
import inspect
import json
import logging
import os
import pickle
import time
import logging
import requests
import warnings

import ccxt
from ccxt.base.errors import NetworkError
# Then you can use it as NetworkError in your code

import numpy as np
import pandas as pd

import requests
from dotenv import load_dotenv
from websocket import create_connection

import config
from pocket_option_client import PocketOptionClient

warnings.filterwarnings(action="ignore", message="datetime.datetime.now(datetime.timezone.utc)")
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[logging.StreamHandler()]
)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s', handlers=[logging.StreamHandler()])


class ProbabilityBrain:
    def __init__(self):
        self.mean = None
        self.std = None
        self.weights = None
        self.bias = 0.0

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y, l2=0.1):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0
        Xn = (X - self.mean) / self.std
        X_aug = np.c_[np.ones((Xn.shape[0], 1)), Xn]
        eye = np.eye(X_aug.shape[1])
        eye[0, 0] = 0.0
        params = np.linalg.solve(X_aug.T @ X_aug + l2 * eye, X_aug.T @ y)
        self.bias = float(params[0])
        self.weights = params[1:]

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        Xn = (X - self.mean) / self.std
        p1 = self._sigmoid(Xn @ self.weights + self.bias)
        return np.c_[1 - p1, p1]

    def save(self, path):
        with open(path, 'wb') as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path):
        with open(path, 'rb') as fh:
            return pickle.load(fh)


class PocketOptionBot:
    FEATURE_COLUMNS = ['rsi', 'ema_fast', 'ema_slow', 'atr', 'volume']
    BASE_OHLCV_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    def __init__(self):
        self.email = config.PO_EMAIL
        self.password = config.PO_PASSWORD
        self.demo_mode = config.DEMO_MODE
        self.risk_pct = config.RISK_PERCENTAGE
        self.api_wss = config.PO_API_WSS
        self.max_trades = config.MAX_TRADES_PER_DAY
        self.min_signal_score = config.MIN_SIGNAL_SCORE
        self.max_daily_drawdown_pct = config.MAX_DAILY_DRAWDOWN_PCT
        self.max_trades_per_day = config.MAX_TRADES_PER_DAY
        self.trade_cooldown_minutes = config.TRADE_COOLDOWN_MINUTES
        self.market_data_limit = config.MARKET_DATA_LIMIT
        self.max_account_use_per_trade_pct = config.MAX_ACCOUNT_USE_PER_TRADE_PCT
        self.stop_loss_stake_pct = config.STOP_LOSS_STAKE_PCT
        self.analysis_timeout_minutes = config.ANALYSIS_TIMEOUT_MINUTES
        self.training_phase = config.TRAINING_PHASE

        self.enable_ai_filter = config.ENABLE_AI_FILTER
        self.ai_model_path = config.AI_MODEL_PATH
        self.trade_memory_path = config.TRADE_MEMORY_PATH
        self.trade_journal_path = config.TRADE_JOURNAL_PATH
        self.ai_min_buy_prob = config.AI_MIN_BUY_PROB
        self.ai_max_sell_prob = config.AI_MAX_SELL_PROB
        self.ai_retrain_every_n_trades = config.AI_RETRAIN_EVERY_N_TRADES
        self.ai_min_retrain_rows = config.AI_MIN_RETRAIN_ROWS

        self.strategy_knowledge_path = config.STRATEGY_KNOWLEDGE_PATH
        self.strategy_review_state_path = config.STRATEGY_REVIEW_STATE_PATH

        self.pairs = config.PAIRS
        self.timeframe = config.TIMEFRAME
        self.balance = config.START_BALANCE
        self.start_of_day_balance = self.balance
        self.daily_trade_count = 0
        self.today = datetime.date.today()
        self.ws = None  # Will hold the websocket connection
        self.last_trade_time = None
        self.model = self.load_ai_model()
        self.market_exchange_ids = config.MARKET_DATA_EXCHANGES
        # Look for this section in your __init__
        self.exchanges = [getattr(ccxt, ex_id)() for ex_id in self.market_exchange_ids if hasattr(ccxt, ex_id)]
        if not self.exchanges:
            self.exchanges = [ccxt.kraken({'enableRateLimit': True})] # Changed from binance

        self.po_client = None
        if config.PO_BASE_URL and config.PO_API_TOKEN:
            self.po_client = PocketOptionClient(config.PO_BASE_URL, config.PO_API_TOKEN)

    @property
    def phase_trade_cap(self):
        if self.training_phase == 'month1':
            return 100
        if self.training_phase == 'month2':
            return 30
        return 3

    @staticmethod
    def market_candidates(symbol):
        mapping = {
            'EURUSD': ['EUR/USDT', 'EUR/USD'],
            'GBPUSD': ['GBP/USDT', 'GBP/USD'],
        }
        if symbol not in mapping:
            raise ValueError(f"Unknown pair for demo data: {symbol}")
        return mapping[symbol]

    def sync_balance_from_pocket_option(self):
        if self.po_client:
            try:
                bal = self.po_client.get_balance(mode=config.PO_ACCOUNT_MODE)
                self.balance = float(bal)
                self.start_of_day_balance = self.balance
                logging.info("Synced Pocket Option %s balance: $%.2f", config.PO_ACCOUNT_MODE, self.balance)
                return
            except Exception as exc:
                logging.warning("Could not sync Pocket Option balance: %s", exc)

        if not config.PO_BALANCE_API:
            return
        try:
            resp = requests.get(config.PO_BALANCE_API, timeout=8)
            resp.raise_for_status()
            payload = resp.json()
            if 'balance' in payload:
                self.balance = float(payload['balance'])
                self.start_of_day_balance = self.balance
                logging.info("Synced fallback balance endpoint: $%.2f", self.balance)
        except Exception as exc:
            logging.warning("Could not sync fallback balance endpoint: %s", exc)

    def load_ai_model(self):
        if not self.enable_ai_filter:
            return None
        if os.path.exists(self.ai_model_path):
            return ProbabilityBrain.load(self.ai_model_path)
        logging.warning("AI model not found at %s. Using rule-only mode.", self.ai_model_path)
        return None

    def reset_daily_trades(self):
        if datetime.date.today() != self.today:
            self.daily_trade_count = 0
            self.today = datetime.date.today()
            self.start_of_day_balance = self.balance
            self.last_trade_time = None

    def ensure_daily_strategy_review(self):
        if not os.path.exists(self.strategy_knowledge_path):
            logging.warning("No strategy knowledge JSON found at %s", self.strategy_knowledge_path)
            return False

        today = datetime.date.today().isoformat()
        state = {}
        if os.path.exists(self.strategy_review_state_path):
            try:
                with open(self.strategy_review_state_path, 'r', encoding='utf-8') as fh:
                    state = json.load(fh)
            except Exception:
                state = {}

        if state.get('last_review_date') == today:
            return True

        start = datetime.datetime.now(datetime.timezone.utc)
        with open(self.strategy_knowledge_path, 'r', encoding='utf-8') as fh:
            knowledge = json.load(fh)

        strategy_count = len(knowledge.get('strategy_items', []))
        video_count = len(knowledge.get('video_sources', []))
        doc_count = len(knowledge.get('document_sources', []))

        time.sleep(min(2, self.analysis_timeout_minutes * 0.1))
        elapsed = datetime.datetime.now(datetime.timezone.utc) - start
        if elapsed > datetime.timedelta(minutes=self.analysis_timeout_minutes):
            logging.warning("Strategy review exceeded max analysis time.")
            return False

        state = {
            'last_review_date': today,
            'reviewed_strategies': strategy_count,
            'reviewed_videos': video_count,
            'reviewed_docs': doc_count,
        }
        with open(self.strategy_review_state_path, 'w', encoding='utf-8') as fh:
            json.dump(state, fh, indent=2)

        logging.info(
            "Daily strategy review complete: strategies=%s videos=%s docs=%s",
            strategy_count,
            video_count,
            doc_count,
        )
        return True

    def get_market_data(self, symbol, timeframe='1m', limit=240):
        last_exc = None
        for exchange in self.exchanges:
            try:
                markets = exchange.load_markets()
                market = next((m for m in self.market_candidates(symbol) if m in markets), None)
                if market is None:
                    continue
                candles = exchange.fetch_ohlcv(market, timeframe=timeframe, limit=limit)
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return self.enrich_market_data(df)
            except NetworkError as exc:
                last_exc = exc
                continue
            except Exception as exc:
                last_exc = exc
                continue
        raise RuntimeError(f"Unable to fetch market data for {symbol} from exchanges={self.market_exchange_ids}: {last_exc}")

    def get_multi_timeframe_snapshot(self, symbol, limit=100):
        snapshots = {}
        for tf in config.HIGHER_TIMEFRAMES:
            snapshots[tf] = self.fetch_market_data_compat(symbol, timeframe=tf, limit=limit)
        return snapshots

    def fetch_market_data_compat(self, symbol, timeframe='1m', limit=240):
        params = inspect.signature(self.get_market_data).parameters
        if 'timeframe' in params:
            raw = self.get_market_data(symbol, timeframe=timeframe, limit=limit)
        else:
            logging.warning("Detected legacy get_market_data signature; ignoring timeframe=%s for %s", timeframe, symbol)
            raw = self.get_market_data(symbol, limit=limit)
        return self.normalize_market_data(raw)

    def normalize_market_data(self, raw_df):
        if raw_df is None:
            return None

        df = raw_df.copy()
        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df, columns=self.BASE_OHLCV_COLUMNS)
            except Exception:
                return None

        required_base = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
        if not required_base.issubset(df.columns):
            return None
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', unit='ms')
        if {'ema_fast', 'ema_slow', 'rsi', 'atr'}.issubset(df.columns):
            return df
        return self.enrich_market_data(df)

    def enrich_market_data(self, df):
        out = df.copy()
        out['ema_fast'] = out['close'].ewm(span=20, adjust=False).mean()
        out['ema_slow'] = out['close'].ewm(span=50, adjust=False).mean()
        delta = out['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        out['rsi'] = 100 - (100 / (1 + rs))
        tr = pd.concat([
            out['high'] - out['low'],
            (out['high'] - out['close'].shift(1)).abs(),
            (out['low'] - out['close'].shift(1)).abs(),
        ], axis=1)
        out['atr'] = tr.max(axis=1).rolling(14).mean()
        out['vol_ma'] = out['volume'].rolling(20).mean()
        return out

    def _ai_probability_up(self, latest_row):
        if self.model is None:
            return None
        vals = latest_row[self.FEATURE_COLUMNS]
        if vals.isna().any():
            return None
        return float(self.model.predict_proba(vals.values.astype(float).reshape(1, -1))[0][1])

    @staticmethod
    def infer_market_state(df):
        if df is None or len(df) == 0:
            return 'Unknown'
        required = {'high', 'low', 'close', 'ema_fast', 'ema_slow'}
        if not required.issubset(df.columns):
            return 'Unknown'
        volatility = (df['high'] - df['low']).tail(30).mean() / max(df['close'].tail(30).mean(), 1e-9)
        trend = abs(df['ema_fast'].iloc[-1] - df['ema_slow'].iloc[-1]) / max(df['close'].iloc[-1], 1e-9)
        if volatility > 0.01:
            return 'Volatile'
        if trend > 0.002:
            return 'Trending'
        return 'Ranging'

    def timeframe_bias(self, snapshot):
        up = 0
        down = 0
        for _, df in snapshot.items():
            if df['ema_fast'].iloc[-1] > df['ema_slow'].iloc[-1]:
                up += 1
            elif df['ema_fast'].iloc[-1] < df['ema_slow'].iloc[-1]:
                down += 1
        if up > down:
            return 'buy'
        if down > up:
            return 'sell'
        return None

    def analyze_ict(self, df, mtf_bias=None, log_signal=True):
        if df is None or len(df) == 0:
            logging.warning("analyze_ict received empty market data; skipping signal evaluation.")
            return None
        required_cols = {'high', 'low', 'close', 'ema_fast', 'ema_slow', 'rsi'}
        if not required_cols.issubset(df.columns):
            logging.warning("analyze_ict missing required columns (%s); got=%s", sorted(required_cols), list(df.columns))
            return None

    def fetch_account_balance(self):
        # In real application, fetch via Pocket Option API (or WebSocket/HTTP endpoint if available)
        # For now, DEMO ONLY!
        logging.info(f"Simulated balance check: ${self.balance:.2f}")
        return self.balance
        work = df.copy()
        work['s_high'] = (work['high'].shift(2) < work['high'].shift(1)) & (work['high'].shift(1) > work['high'])
        work['s_low'] = (work['low'].shift(2) > work['low'].shift(1)) & (work['low'].shift(1) < work['low'])
        work['fvg_up'] = work['low'] > work['high'].shift(2)
        work['fvg_down'] = work['high'] < work['low'].shift(2)
        latest = work.iloc[-1]

        bullish_score = int(any(work['s_low'].tail(5))) * 2 + int(any(work['fvg_up'].tail(3))) * 2
        bullish_score += int(latest['ema_fast'] > latest['ema_slow']) * 2 + int(45 <= latest['rsi'] <= 65)
        bearish_score = int(any(work['s_high'].tail(5))) * 2 + int(any(work['fvg_down'].tail(3))) * 2
        bearish_score += int(latest['ema_fast'] < latest['ema_slow']) * 2 + int(35 <= latest['rsi'] <= 55)

        direction = None
        if bullish_score >= self.min_signal_score and bullish_score > bearish_score:
            direction = 'buy'
        elif bearish_score >= self.min_signal_score and bearish_score > bullish_score:
            direction = 'sell'
        if direction is None:
            return None

        if mtf_bias and direction != mtf_bias:
            return None

        prob_up = self._ai_probability_up(latest)
        if prob_up is not None:
            if direction == 'buy' and prob_up < self.ai_min_buy_prob:
                return None
            if direction == 'sell' and prob_up > self.ai_max_sell_prob:
                return None

        if log_signal:
            logging.info("Signal=%s bull=%s bear=%s mtf_bias=%s ai_p_up=%s", direction, bullish_score, bearish_score, mtf_bias, prob_up)
        return direction

    def can_trade_by_phase(self):
        return self.daily_trade_count < self.phase_trade_cap

    def risk_size(self, entry, stop):
        risk_amount = self.fetch_account_balance() * (self.risk_pct / 100)
        risk_amount = self.balance * (self.risk_pct / 100)
        risk_per_trade = max(abs(entry - stop), 0.0001)
        size = abs(risk_amount / risk_per_trade)
        return round(size, 6)

    def get_market_data(self, symbol, limit=120):
    
        exchange = ccxt.kraken({'enableRateLimit': True})
    
    # Kraken uses different naming for these pairs
        if symbol == 'EURUSD':
            market = 'EUR/USD'
        elif symbol == 'GBPUSD':
            market = 'GBP/USD'
        else:
            market = symbol.replace('USD', '/USD') if 'USD' in symbol else symbol
        
        df = exchange.fetch_ohlcv(market, timeframe='1m', limit=limit)
        df = pd.DataFrame(df, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df


    def analyze_ict(self, df):
        # ----- MINIMAL ICT/SMART MONEY LOGIC (all-in-one for space) -----
        # 1. Market Structure: find swing highs/lows
        df['s_high'] = (df['high'].shift(2) < df['high'].shift(1)) & (df['high'].shift(1) > df['high'])  # peak
        df['s_low'] = (df['low'].shift(2) > df['low'].shift(1)) & (df['low'].shift(1) < df['low'])

        # 2. FVG scan: identify fair value gaps
        df['fvg_up'] = (df['low'] > df['high'].shift(2))
        df['fvg_down'] = (df['high'] < df['low'].shift(2))

        # 3. Session check (basic): only trade London/NY hours UTC
        now = datetime.datetime.now(datetime.UTC).hour
        session_ok = (7 <= now <= 17)

        # 4. Signal generation:
        bullish = any(df['s_low'][-5:]) and any(df['fvg_up'][-3:]) and session_ok
        bearish = any(df['s_high'][-5:]) and any(df['fvg_down'][-3:]) and session_ok

        signal = None
        if bullish and not bearish:
            signal = 'buy'
        elif bearish and not bullish:
            signal = 'sell'
        return signal

    def ensure_api_freshness(self):
        # Dummy implementation for demonstration. Real trading bots fetch fresh endpoint config from PO docs or via API gateway
        # This could be honed to auto-discover healthiest endpoint.
        logging.info("Checked Pocket Option API endpoint freshness (simulated auto-discovery)")

    def trade(self, pair, direction, size):
        # Simulate placing a trade via API.
        self.fetch_account_balance()  # refresh
        max_size_by_account_use = self.balance * (self.max_account_use_per_trade_pct / 100)
        return round(min(size, max_size_by_account_use), 6)

    def log_trade_memory(self, latest_row, direction, profit, entry_setup, market_state, spread):
        target = 1 if profit > 0 else 0
        file_exists = os.path.isfile(self.trade_memory_path)
        with open(self.trade_memory_path, 'a', newline='') as fh:
            writer = csv.writer(fh)
            if not file_exists:
                writer.writerow(self.FEATURE_COLUMNS + ['direction', 'target', 'market_state', 'entry_setup', 'spread_at_entry'])
            writer.writerow([latest_row[c] for c in self.FEATURE_COLUMNS] + [direction, target, market_state, entry_setup, spread])

    def reward_score(self, profit_pips, spread_pips, hold_minutes):
        return (profit_pips * 10) - (spread_pips * 5) - (hold_minutes / 30)

    def log_trade_journal(self, record):
        fields = [
            'Trade_ID', 'timestamp', 'pair', 'direction', 'entry', 'stop', 'size', 'profit', 'balance_after',
            'source', 'won', 'Market_State', 'Time_of_Day', 'Entry_Setup', 'Spread_at_Entry', 'Hold_Minutes', 'Reward_Score'
        ]
        file_exists = os.path.isfile(self.trade_journal_path)
        with open(self.trade_journal_path, 'a', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)

    def next_trade_id(self):
        if not os.path.exists(self.trade_journal_path):
            return 1
        df = pd.read_csv(self.trade_journal_path)
        if df.empty:
            return 1
        return int(df['Trade_ID'].max()) + 1

    def retrain_ai_from_memory(self):
        if not os.path.exists(self.trade_memory_path):
            return
        memory_df = pd.read_csv(self.trade_memory_path)
        if len(memory_df) < self.ai_min_retrain_rows:
            return
        clean = memory_df.dropna(subset=self.FEATURE_COLUMNS + ['target'])
        if clean.empty:
            return
        X = clean[self.FEATURE_COLUMNS].values
        y = clean['target'].astype(int).values
        self.model = ProbabilityBrain()
        self.model.fit(X, y)
        self.model.save(self.ai_model_path)

    @staticmethod
    def performance_metrics(df):
        trades = len(df)
        wins = int((df['won'] == 1).sum())
        losses = trades - wins
        win_rate = (wins / trades * 100) if trades else 0.0
        gross_profit = float(df.loc[df['profit'] > 0, 'profit'].sum())
        gross_loss = float(-df.loc[df['profit'] < 0, 'profit'].sum())
        pf = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        start_balance = float(df['balance_after'].iloc[0] - df['profit'].iloc[0]) if trades else 0.0
        end_balance = float(df['balance_after'].iloc[-1]) if trades else 0.0
        roi = ((end_balance - start_balance) / start_balance * 100) if start_balance > 0 else 0.0
        eq = df['balance_after'].astype(float)
        mdd = float(((eq - eq.cummax()) / eq.cummax()).min() * 100) if trades else 0.0
        return {'trades': trades, 'wins': wins, 'losses': losses, 'win_rate_pct': round(win_rate, 2), 'profit_factor': round(pf, 3) if np.isfinite(pf) else 'inf', 'roi_pct': round(roi, 2), 'max_drawdown_pct': round(mdd, 2)}

    def report_performance(self, source=None):
        if not os.path.exists(self.trade_journal_path):
            logging.warning("No trade journal found")
            return
        df = pd.read_csv(self.trade_journal_path)
        if source:
            df = df[df['source'] == source]
        if df.empty:
            logging.warning("No rows for source=%s", source)
            return
        stats = self.performance_metrics(df)
        logging.info("PERFORMANCE [%s] %s", source or 'all', stats)

    def apply_month2_filter(self, pair, market_state, hour):
        if self.training_phase == 'month1':
            return True
        if not os.path.exists(self.trade_journal_path):
            return self.training_phase != 'sniper'
        df = pd.read_csv(self.trade_journal_path)
        if df.empty or 'Reward_Score' not in df.columns:
            return self.training_phase != 'sniper'
        top = df.nlargest(max(1, int(len(df) * 0.05)), 'Reward_Score')
        if top.empty:
            return self.training_phase != 'sniper'
        pair_ok = pair in top['pair'].astype(str).unique().tolist()
        state_ok = market_state in top['Market_State'].astype(str).unique().tolist()
        hour_ok = int(hour) in top['Time_of_Day'].astype(int).tolist()
        return pair_ok and state_ok and hour_ok

    def execute_live_order(self, pair, direction, stake):
        if not self.po_client:
            raise RuntimeError('PO client not configured. Set PO_BASE_URL and PO_API_TOKEN.')

        order_id = self.po_client.place_order(
            symbol=pair,
            direction=direction,
            amount=stake,
            duration_sec=config.PO_ORDER_DURATION_SEC,
            mode=config.PO_ACCOUNT_MODE,
        )
        result = self.po_client.wait_for_result(
            order_id,
            poll_interval_sec=config.PO_POLL_INTERVAL_SEC,
            max_wait_sec=max(180, config.PO_ORDER_DURATION_SEC * 4),
        )
        profit = float(result.get('profit', 0.0))
        return profit

    def execute_trade(self, pair, direction, size, latest_row, entry, stop, market_state, entry_setup):
        self.daily_trade_count += 1
        stake = min(size, self.balance * (self.max_account_use_per_trade_pct / 100))
        stop_loss_cap = stake * (self.stop_loss_stake_pct / 100)

        hold_minutes = np.random.randint(1, 15)
        spread_pips = np.random.uniform(0.5, 2.5)

        if self.demo_mode:
            # "Place" the trade and simulate instant result (for demo)
            self.daily_trade_count += 1
            change = np.random.uniform(0.75, 1.15) if direction == 'buy' else np.random.uniform(0.75, 1.15) * -1
            profit = size * change
            self.balance += profit
            outcome = "Profit" if profit > 0 else "Loss"
            logging.info(f"TRADE: {pair} | {direction.upper()} | Size {size} | {outcome} ${abs(profit):.2f} | New Balance: ${self.balance:.2f}")
            pnl_raw = stake * np.random.normal(loc=0.05, scale=0.35)
            profit = max(pnl_raw, -stop_loss_cap)
        else:
            # Here you integrate a real API call to Pocket Option's live websocket (not demo)
            logging.info("LIVE trade function NOT IMPLEMENTED in this demo bot.")
            profit = self.execute_live_order(pair, direction, stake)
            profit = max(profit, -stop_loss_cap)

        self.balance += profit
        self.last_trade_time = datetime.datetime.now(datetime.timezone.utc)

        profit_pips = abs(profit) / max(stake, 1e-9) * 100
        reward = self.reward_score(profit_pips, spread_pips, hold_minutes)

        self.log_trade_memory(latest_row, direction, profit, entry_setup, market_state, spread_pips)
        self.log_trade_journal({
            'Trade_ID': self.next_trade_id(),
            'timestamp': self.last_trade_time.isoformat(),
            'pair': pair,
            'direction': direction,
            'entry': round(float(entry), 8),
            'stop': round(float(stop), 8),
            'size': round(float(stake), 6),
            'profit': round(float(profit), 6),
            'balance_after': round(float(self.balance), 6),
            'source': 'live_demo' if self.demo_mode else 'live',
            'won': 1 if profit > 0 else 0,
            'Market_State': market_state,
            'Time_of_Day': self.last_trade_time.hour,
            'Entry_Setup': entry_setup,
            'Spread_at_Entry': round(float(spread_pips), 4),
            'Hold_Minutes': int(hold_minutes),
            'Reward_Score': round(float(reward), 4),
        })

        if self.ai_retrain_every_n_trades > 0 and self.daily_trade_count % self.ai_retrain_every_n_trades == 0:
            self.retrain_ai_from_memory()

        logging.info("TRADE %s %s stake=%.2f pnl=%.2f bal=%.2f reward=%.2f", pair, direction.upper(), stake, profit, self.balance, reward)

    def run_backtest(self, bars=800, payout=0.82):
        for pair in self.pairs:
            try:
                df = self.fetch_market_data_compat(pair, timeframe='1m', limit=bars)
            except RuntimeError as exc:
                logging.warning("Skipping %s backtest: %s", pair, exc)
                continue
            equity = self.start_of_day_balance
            rows = []
            for i in range(100, len(df) - 1):
                window = df.iloc[:i + 1]
                mtf = self.get_multi_timeframe_snapshot(pair, limit=100)
                direction = self.analyze_ict(window, mtf_bias=self.timeframe_bias(mtf), log_signal=False)
                if not direction:
                    continue
                entry = float(df['close'].iloc[i])
                nxt = float(df['close'].iloc[i + 1])
                risk_amount = equity * (self.risk_pct / 100)
                won = (direction == 'buy' and nxt > entry) or (direction == 'sell' and nxt < entry)
                profit = risk_amount * payout if won else -risk_amount
                equity += profit
                hold = 1
                spread = 1.0
                reward = self.reward_score(abs(nxt - entry) * 10000, spread, hold)
                rows.append({'Trade_ID': self.next_trade_id(), 'timestamp': df['timestamp'].iloc[i + 1].isoformat(), 'pair': pair, 'direction': direction, 'entry': entry, 'stop': np.nan, 'size': risk_amount, 'profit': profit, 'balance_after': equity, 'source': 'backtest', 'won': 1 if won else 0, 'Market_State': self.infer_market_state(window), 'Time_of_Day': int(df['timestamp'].iloc[i + 1].hour), 'Entry_Setup': 'ICT_MTF', 'Spread_at_Entry': spread, 'Hold_Minutes': hold, 'Reward_Score': reward})
            for rec in rows:
                self.log_trade_journal(rec)
            if rows:
                stats = self.performance_metrics(pd.DataFrame(rows))
                logging.info("%s backtest %s", pair, stats)

    def run(self):
        logging.info("=== Pocket Option ICT 5-Star Bot (Upgraded Edition) ===")
        self.sync_balance_from_pocket_option()
        while True:
            self.reset_daily_trades()
            if self.daily_trade_count >= self.max_trades:
                logging.info(f"Trade cap hit for today ({self.max_trades} trades). Sleeping until tomorrow.")
                time.sleep(60 * 60 * 2)

            if self.daily_trade_count >= self.max_trades_per_day:
                logging.info("Daily trade cap reached (%s). Sleeping before next cycle.", self.max_trades_per_day)
                time.sleep(60 * 30)
                continue

            if not self.ensure_daily_strategy_review():
                logging.warning("Strategy review not completed; waiting.")
                time.sleep(60)
                continue
            self.ensure_api_freshness()

            if not self.can_trade_by_phase():
                logging.info("Phase trade cap reached (%s).", self.phase_trade_cap)
                time.sleep(60 * 30)
                continue

            if ((self.start_of_day_balance - self.balance) / max(self.start_of_day_balance, 1e-9)) * 100 >= self.max_daily_drawdown_pct:
                logging.warning("Daily drawdown cap reached.")
                time.sleep(60 * 30)
                continue

            if self.last_trade_time is not None:
                elapsed = datetime.datetime.now(datetime.timezone.utc) - self.last_trade_time
                if elapsed < datetime.timedelta(minutes=self.trade_cooldown_minutes):
                    time.sleep(60)
                    continue

            for pair in self.pairs:
                df = self.get_market_data(pair)
                direction = self.analyze_ict(df)
                if direction:
                    entry = df['close'].iloc[-1]
                    # Stop is 1 ATR below/above for risk calculation (simplified)
                    atr = df['high'].rolling(14).max().iloc[-1] - df['low'].rolling(14).min().iloc[-1]
                    stop = entry - atr if direction == 'buy' else entry + atr
                    size = self.risk_size(entry, stop)
                    if self.daily_trade_count < self.max_trades:
                        self.trade(pair, direction, size)
                
                if self.daily_trade_count >= self.max_trades_per_day:
                    logging.info("Daily trade cap reached (%s).", self.max_trades_per_day)
                    break
                time.sleep(120)  # align with 2min candle (demo frequency)

                try:
                    df = self.fetch_market_data_compat(pair, timeframe='1m', limit=max(120, self.market_data_limit))
                    mtf = self.get_multi_timeframe_snapshot(pair, limit=100)
                except RuntimeError as exc:
                    logging.warning("Skipping %s: %s", pair, exc)
                    continue
                if df is None or len(df) == 0:
                    logging.warning("Skipping %s: empty market data returned.", pair)
                    continue

                market_state = self.infer_market_state(df)
                hour = int(df['timestamp'].iloc[-1].hour)
                if not self.apply_month2_filter(pair, market_state, hour):
                    continue

                mtf_bias = self.timeframe_bias(mtf)
                direction = self.analyze_ict(df, mtf_bias=mtf_bias)
                if not direction:
                    continue

                analysis_start = datetime.datetime.now(datetime.timezone.utc)
                entry = float(df['close'].iloc[-1])
                atr = float(df['atr'].iloc[-1])
                if np.isnan(atr) or atr <= 0:
                    continue
                stop = entry - atr if direction == 'buy' else entry + atr
                size = self.risk_size(entry, stop)
                entry_setup = 'ICT_MTF_BIAS_AI'

                if datetime.datetime.now(datetime.timezone.utc) - analysis_start > datetime.timedelta(minutes=self.analysis_timeout_minutes):
                    logging.warning("Analysis exceeded timeout, skipping trade.")
                    continue

                self.execute_trade(pair, direction, size, df.iloc[-1], entry, stop, market_state, entry_setup)
                if not self.can_trade_by_phase():
                    break

            time.sleep(120)


def parse_args():
    parser = argparse.ArgumentParser(description="Pocket Option ICT + AI bot")
    parser.add_argument('--mode', choices=['run', 'backtest', 'report'], default='run')
    parser.add_argument('--bars', type=int, default=800)
    parser.add_argument('--payout', type=float, default=0.82)
    parser.add_argument('--source', choices=['all', 'live_demo', 'live', 'backtest'], default='all')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    bot = PocketOptionBot()
    bot.run()
    if args.mode == 'backtest':
        bot.run_backtest(bars=args.bars, payout=args.payout)
    elif args.mode == 'report':
        bot.report_performance(source=None if args.source == 'all' else args.source)
    else:
        bot.run()