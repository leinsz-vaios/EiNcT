"""
POCKET OPTION ICT BOT - HYBRID EDITION
Combines ICT Unified Protocol (Gates) + Scoring System + AI Learning
"""
import warnings
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

import ccxt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

import config
from pocket_option_client import PocketOptionClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[logging.StreamHandler()]
)


class ProbabilityBrain:
    """AI model for learning from past trades"""
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
        # Account settings
        self.email = config.PO_EMAIL
        self.password = config.PO_PASSWORD
        self.demo_mode = config.DEMO_MODE
        self.balance = config.START_BALANCE
        self.start_of_day_balance = self.balance
        
        # Trading settings
        self.pairs = config.PAIRS
        self.timeframe = config.TIMEFRAME
        self.risk_pct = config.RISK_PERCENTAGE
        self.max_trades_per_day = config.MAX_TRADES_PER_DAY
        self.max_daily_drawdown_pct = config.MAX_DAILY_DRAWDOWN_PCT
        self.trade_cooldown_minutes = config.TRADE_COOLDOWN_MINUTES
        self.max_account_use_per_trade_pct = config.MAX_ACCOUNT_USE_PER_TRADE_PCT
        self.stop_loss_stake_pct = config.STOP_LOSS_STAKE_PCT
        self.training_phase = config.TRAINING_PHASE
        
        # ICT Gate settings
        self.enable_time_filter = config.ENABLE_TIME_FILTER
        self.london_killzone = config.LONDON_KILLZONE_UTC
        self.ny_killzone = config.NY_KILLZONE_UTC
        self.liquidity_lookback = config.LIQUIDITY_LOOKBACK
        self.sweep_memory = config.SWEEP_MEMORY_CANDLES
        
        # Scoring settings
        self.min_signal_score = config.MIN_SIGNAL_SCORE
        
        # AI settings
        self.enable_ai_filter = config.ENABLE_AI_FILTER
        self.ai_model_path = config.AI_MODEL_PATH
        self.trade_memory_path = config.TRADE_MEMORY_PATH
        self.trade_journal_path = config.TRADE_JOURNAL_PATH
        self.ai_min_buy_prob = config.AI_MIN_BUY_PROB
        self.ai_max_sell_prob = config.AI_MAX_SELL_PROB
        self.ai_retrain_every_n_trades = config.AI_RETRAIN_EVERY_N_TRADES
        self.ai_min_retrain_rows = config.AI_MIN_RETRAIN_ROWS
        
        # State tracking
        self.daily_trade_count = 0
        self.today = datetime.date.today()
        self.last_trade_time = None
        self.model = self.load_ai_model()
        self.signal_weights_path = "signal_weights.json"
        self.last_signal_profile = None
        self.signal_weights = self.load_signal_weights()
        
        # Market data
        self.market_exchange_ids = config.MARKET_DATA_EXCHANGES
        self.exchanges = [getattr(ccxt, ex_id)() for ex_id in self.market_exchange_ids if hasattr(ccxt, ex_id)]
        if not self.exchanges:
            self.exchanges = [ccxt.kraken({'enableRateLimit': True})]
        
        # Pocket Option client
        self.po_client = None
        if config.PO_BASE_URL and config.PO_API_TOKEN:
            self.po_client = PocketOptionClient(config.PO_BASE_URL, config.PO_API_TOKEN)
        
        logging.info("=" * 60)
        logging.info("POCKET OPTION ICT BOT - HYBRID EDITION")
        logging.info("=" * 60)
        logging.info("Mode: %s", "DEMO" if self.demo_mode else "LIVE")
        logging.info("Balance: $%.2f", self.balance)
        logging.info("Pairs: %s", ", ".join(self.pairs))
        logging.info("Training Phase: %s (Max %d trades/day)", self.training_phase, self.phase_trade_cap)
        logging.info("=" * 60)

    @property
    def phase_trade_cap(self):
        """Progressive learning caps"""
        if self.training_phase == 'month1':
            return 100
        if self.training_phase == 'month2':
            return 30
        return 3

    def is_crypto(self, pair):
        return pair in ['ETHUSD', 'BTCUSD']

    def load_ai_model(self):
        if not self.enable_ai_filter:
            return None
        if os.path.exists(self.ai_model_path):
            model = ProbabilityBrain.load(self.ai_model_path)
            logging.info("Loaded AI model from %s", self.ai_model_path)
            return model
        logging.warning("AI model not found. Starting fresh.")
        return None

    def load_signal_weights(self):
        """Load adaptive signal weights"""
        defaults = {
            "liq_sweep_buy": 5.0,      # Strongest signal
            "liq_sweep_sell": 5.0,
            "displacement_up": 3.0,
            "displacement_down": 3.0,
            "structure_break_up": 3.0,
            "structure_break_down": 3.0,
            "fvg_up": 1.5,
            "fvg_down": 1.5,
            "mtf_bias": 2.0,
            "rsi_zone": 1.0,
        }
        if os.path.exists(self.signal_weights_path):
            try:
                with open(self.signal_weights_path, "r") as fh:
                    saved = json.load(fh)
                for key, value in saved.items():
                    if key in defaults:
                        defaults[key] = float(value)
                logging.info("Loaded signal weights from %s", self.signal_weights_path)
            except Exception as exc:
                logging.warning("Could not load signal weights: %s", exc)
        return defaults

    def save_signal_weights(self):
        try:
            with open(self.signal_weights_path, "w") as fh:
                json.dump(self.signal_weights, fh, indent=2)
        except Exception as exc:
            logging.warning("Could not save signal weights: %s", exc)

    def update_signal_weights(self, profile, won):
        """Update weights based on trade outcome (slower adaptation to reduce overfitting)"""
        if not profile:
            return
        step = 0.05 if won else -0.025
        for name, triggered in profile.items():
            if not triggered:
                continue
            current = float(self.signal_weights.get(name, 1.0))
            current += step
            self.signal_weights[name] = float(np.clip(current, 0.5, 6.0))
        self.save_signal_weights()

    def reset_daily_trades(self):
        if datetime.date.today() != self.today:
            logging.info("=" * 60)
            logging.info("NEW DAY - Resetting counters")
            logging.info("Yesterday's P/L: $%.2f", self.balance - self.start_of_day_balance)
            logging.info("=" * 60)
            self.daily_trade_count = 0
            self.today = datetime.date.today()
            self.start_of_day_balance = self.balance
            self.last_trade_time = None

    def get_market_data(self, symbol, timeframe='1m', limit=120):
        exchange = self.exchanges[0]

        if symbol == 'EURUSD':
            market = 'EUR/USD'
        elif symbol == 'GBPUSD':
            market = 'GBP/USD'
        elif symbol == 'ETHUSD':
            market = 'ETH/USD'
        else:
            raise Exception(f"Unknown pair: {symbol}")

        candles = exchange.fetch_ohlcv(market, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(candles, columns=self.BASE_OHLCV_COLUMNS)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return self.enrich_market_data(df)

    def enrich_market_data(self, df):
        """Add technical indicators"""
        out = df.copy()
        out['ema_fast'] = out['close'].ewm(span=20, adjust=False).mean()
        out['ema_slow'] = out['close'].ewm(span=50, adjust=False).mean()
        
        # RSI
        delta = out['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        out['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR
        tr = pd.concat([
            out['high'] - out['low'],
            (out['high'] - out['close'].shift(1)).abs(),
            (out['low'] - out['close'].shift(1)).abs(),
        ], axis=1)
        out['atr'] = tr.max(axis=1).rolling(14).mean()
        out['vol_ma'] = out['volume'].rolling(20).mean()
        
        return out

    def get_multi_timeframe_snapshot(self, symbol, limit=100):
        """Get data from higher timeframes for bias"""
        snapshots = {}
        for tf in config.HIGHER_TIMEFRAMES:
            snapshots[tf] = self.get_market_data(symbol, timeframe=tf, limit=limit)
        return snapshots

    def timeframe_bias(self, snapshot):
        """Determine higher timeframe bias"""
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

    def _ai_probability_up(self, latest_row):
        """Get AI prediction"""
        if self.model is None:
            return None
        vals = latest_row[self.FEATURE_COLUMNS]
        if vals.isna().any():
            return None
        return float(self.model.predict_proba(vals.values.astype(float).reshape(1, -1))[0][1])

    def analyze_ict_hybrid(self, df, pair=None, mtf_bias=None, log_signal=True):
        """
        HYBRID ICT ANALYSIS
        Combines: ICT Gates (filters) + Scoring System (selection) + AI (confirmation)
        
        Flow:
        1. GATE 1: Time Filter (Silver Bullet)
        2. GATE 2: Liquidity Sweep Detection (Liquidity Purge + Turtle Soup)
        3. GATE 3: Structure Break (Smart Money Reversal)
        4. SCORING: If passes all gates, score the setup
        5. AI FILTER: Final probability check
        """
        if df is None or len(df) < 50:
            return None
        
        required_cols = {'open', 'high', 'low', 'close', 'ema_fast', 'ema_slow', 'rsi', 'atr', 'timestamp'}
        if not required_cols.issubset(df.columns):
            return None
        
        work = df.copy()
        latest = work.iloc[-1]
        
        latest_time = pd.to_datetime(latest['timestamp'])
        hour_utc = latest_time.hour
        
        # --- ADD IT HERE ---
        print(f"DEBUG: Processing {pair} at {hour_utc}:00 UTC... Filter active: {self.enable_time_filter}")
        # --------------------
        
        # ================================================================
        # GATE 1: SILVER BULLET (Time Filter)
        # ================================================================
        if self.enable_time_filter:
            in_london = self.london_killzone[0] <= hour_utc < self.london_killzone[1]
            in_ny = self.ny_killzone[0] <= hour_utc < self.ny_killzone[1]
            in_killzone = in_london or in_ny
            
            if not in_killzone:
                return None  # Not in trading window
        
        # ================================================================
        # DATA PREPARATION: Identify Liquidity Pools & Structure
        # ================================================================
        prev_high = work['high'].rolling(self.liquidity_lookback).max().shift(1)
        prev_low = work['low'].rolling(self.liquidity_lookback).min().shift(1)
        
        # Liquidity Sweeps (Purge + Rejection = Turtle Soup)
        work['liq_sweep_buy'] = (work['low'] < prev_low) & (work['close'] > prev_low)
        work['liq_sweep_sell'] = (work['high'] > prev_high) & (work['close'] < prev_high)
        
        # Displacement
        work['body'] = (work['close'] - work['open']).abs()
        work['displacement'] = work['body'] > (work['atr'] * 0.8)
        
        # Structure Breaks
        work['structure_break_up'] = work['close'] > work['high'].shift(5).rolling(5).max()
        work['structure_break_down'] = work['close'] < work['low'].shift(5).rolling(5).min()
        
        # Fair Value Gaps
        work['fvg_up'] = work['low'] > work['high'].shift(2)
        work['fvg_down'] = work['high'] < work['low'].shift(2)
        
        latest = work.iloc[-1]
        
        # ================================================================
        # GATE 2 & 3: LIQUIDITY SWEEP + STRUCTURE BREAK
        # ================================================================
        # Check if sweep happened recently (not just current candle)
        recent_sweep_buy = any(work['liq_sweep_buy'].tail(self.sweep_memory))
        recent_sweep_sell = any(work['liq_sweep_sell'].tail(self.sweep_memory))
        
        # Current candle must show structure break + displacement
        current_displacement_up = latest['displacement'] and latest['close'] > latest['open']
        current_displacement_down = latest['displacement'] and latest['close'] < latest['open']
        current_structure_up = latest['structure_break_up']
        current_structure_down = latest['structure_break_down']
        
        # Determine if we have a valid ICT setup
        bullish_setup = recent_sweep_buy and current_displacement_up and current_structure_up
        bearish_setup = recent_sweep_sell and current_displacement_down and current_structure_down
        
        if not (bullish_setup or bearish_setup):
            return None  # Didn't pass ICT gates
        
        # ================================================================
        # SCORING SYSTEM: Score the setup quality
        # ================================================================
        is_crypto = self.is_crypto(pair) if pair else False
        
        # RSI zones
        if is_crypto:
            rsi_buy_zone = 50 <= latest["rsi"] <= 70
            rsi_sell_zone = 30 <= latest["rsi"] <= 50
        else:
            rsi_buy_zone = 45 <= latest["rsi"] <= 65
            rsi_sell_zone = 35 <= latest["rsi"] <= 55
        
        # Build signal profile
        signals = {
            "liq_sweep_buy": bullish_setup,
            "liq_sweep_sell": bearish_setup,
            "displacement_up": current_displacement_up,
            "displacement_down": current_displacement_down,
            "structure_break_up": current_structure_up,
            "structure_break_down": current_structure_down,
            "fvg_up": bool(any(work["fvg_up"].tail(3))),
            "fvg_down": bool(any(work["fvg_down"].tail(3))),
            "mtf_bias": mtf_bias is not None,
            "rsi_zone": False,
        }
        
        buy_score = 0.0
        sell_score = 0.0
        
        # Calculate scores
        if signals["liq_sweep_buy"]:
            buy_score += self.signal_weights["liq_sweep_buy"]
        if signals["liq_sweep_sell"]:
            sell_score += self.signal_weights["liq_sweep_sell"]
        if signals["displacement_up"]:
            buy_score += self.signal_weights["displacement_up"]
        if signals["displacement_down"]:
            sell_score += self.signal_weights["displacement_down"]
        if signals["structure_break_up"]:
            buy_score += self.signal_weights["structure_break_up"]
        if signals["structure_break_down"]:
            sell_score += self.signal_weights["structure_break_down"]
        if signals["fvg_up"]:
            buy_score += self.signal_weights["fvg_up"]
        if signals["fvg_down"]:
            sell_score += self.signal_weights["fvg_down"]
        
        # MTF Bias (filters wrong direction)
        if mtf_bias == "buy":
            buy_score += self.signal_weights["mtf_bias"]
            sell_score = 0  # Filter out sells
        elif mtf_bias == "sell":
            sell_score += self.signal_weights["mtf_bias"]
            buy_score = 0  # Filter out buys
        
        # RSI Zone
        if rsi_buy_zone:
            buy_score += self.signal_weights["rsi_zone"]
            signals["rsi_zone"] = True
        if rsi_sell_zone:
            sell_score += self.signal_weights["rsi_zone"]
            signals["rsi_zone"] = True
        
        # Determine direction
        direction = None
        min_score = self.min_signal_score
        
        if buy_score >= min_score and buy_score > sell_score:
            direction = 'buy'
        elif sell_score >= min_score and sell_score > buy_score:
            direction = 'sell'
        
        if direction is None:
            return None
        
        # ================================================================
        # AI FILTER: Final confirmation
        # ================================================================
        prob_up = self._ai_probability_up(latest)
        if prob_up is not None:
            if direction == 'buy' and prob_up < self.ai_min_buy_prob:
                return None
            if direction == 'sell' and prob_up > self.ai_max_sell_prob:
                return None
        
        # Store signal profile for learning
        self.last_signal_profile = signals
        
        if log_signal:
            logging.info(
                "✅ SIGNAL: %s | Score: %.1f | MTF: %s | AI Prob: %.2f | Time: %02d:00 UTC",
                direction.upper(),
                buy_score if direction == 'buy' else sell_score,
                mtf_bias or 'neutral',
                prob_up if prob_up else 0.5,
                hour_utc
            )
        
        return direction

    def simulate_trade_outcome(self, df, current_idx, direction, stake, entry, stop):
        """Realistic simulation using actual price movement"""
        lookhead = min(5, len(df) - current_idx - 1)
        if lookhead <= 0:
            won = np.random.random() < 0.5
            profit = stake * 0.82 if won else -stake
            return profit, 5
        
        future_prices = df['close'].iloc[current_idx + 1:current_idx + 1 + lookhead]
        
        if direction == 'buy':
            won = future_prices.max() > entry
        else:
            won = future_prices.min() < entry
        
        profit = stake * 0.82 if won else -stake
        return profit, lookhead

    def execute_trade(self, pair, direction, size, latest_row, entry, stop, market_state, entry_setup, df=None, current_idx=None):
        """Execute trade with proper tracking"""
        self.daily_trade_count += 1
        stake = min(size, self.balance * (self.max_account_use_per_trade_pct / 100))
        
        # Realistic spread
        if self.is_crypto(pair):
            spread_pips = np.random.uniform(2.0, 5.0)
        else:
            spread_pips = np.random.uniform(1.5, 2.5)
        
        if self.demo_mode:
            if df is not None and current_idx is not None:
                profit, hold_minutes = self.simulate_trade_outcome(df, current_idx, direction, stake, entry, stop)
            else:
                won = np.random.random() < 0.5
                profit = stake * 0.82 if won else -stake
                hold_minutes = 5
            
            # Apply costs
            profit -= (spread_pips / 10000) * stake
            profit = max(profit, -stake)  # Can't lose more than stake
        else:
            # Live execution
            try:
                profit = self.execute_live_order(pair, direction, stake)
                hold_minutes = config.PO_ORDER_DURATION_SEC // 60
            except Exception as exc:
                logging.error("Live order failed: %s", exc)
                return
        
        # Update balance
        old_balance = self.balance
        self.balance += profit
        self.last_trade_time = datetime.datetime.now(datetime.timezone.utc)
        
        # Calculate reward
        profit_pct = (profit / stake) * 100
        reward = profit_pct - (spread_pips * 0.5)
        
        # Log trade
        won = 1 if profit > 0 else 0
        self.log_trade_memory(latest_row, direction, profit, entry_setup, market_state, spread_pips)
        self.log_trade_journal({
            'Trade_ID': self.next_trade_id(),
            'timestamp': self.last_trade_time.isoformat(),
            'pair': pair,
            'direction': direction,
            'entry': round(float(entry), 8),
            'stop': round(float(stop), 8),
            'size': round(float(stake), 2),
            'profit': round(float(profit), 2),
            'balance_after': round(float(self.balance), 2),
            'source': 'demo' if self.demo_mode else 'live',
            'won': won,
            'Market_State': market_state,
            'Time_of_Day': self.last_trade_time.hour,
            'Entry_Setup': entry_setup,
            'Spread_at_Entry': round(float(spread_pips), 2),
            'Hold_Minutes': int(hold_minutes),
            'Reward_Score': round(float(reward), 2),
        })
        
        # Update learning
        if self.ai_retrain_every_n_trades > 0 and self.daily_trade_count % self.ai_retrain_every_n_trades == 0:
            self.retrain_ai_from_memory()
        
        self.update_signal_weights(self.last_signal_profile, profit > 0)
        
        # Log result
        result = "WIN" if won else "LOSS"
        pnl_display = f"+${profit:.2f}" if profit > 0 else f"-${abs(profit):.2f}"
        logging.info(
            "🎯 TRADE #%d: %s %s | Stake: $%.2f | %s %s | Balance: $%.2f → $%.2f (Daily: %+.2f%%)",
            self.daily_trade_count,
            pair,
            direction.upper(),
            stake,
            result,
            pnl_display,
            old_balance,
            self.balance,
            ((self.balance - self.start_of_day_balance) / self.start_of_day_balance * 100)
        )

    def execute_live_order(self, pair, direction, stake):
        """Execute live order via Pocket Option API"""
        if not self.po_client:
            raise RuntimeError('PO client not configured')
        
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
        return float(result.get('profit', 0.0))

    def log_trade_memory(self, latest_row, direction, profit, entry_setup, market_state, spread):
        """Log trade for AI learning"""
        target = 1 if profit > 0 else 0
        file_exists = os.path.isfile(self.trade_memory_path)
        with open(self.trade_memory_path, 'a', newline='') as fh:
            writer = csv.writer(fh)
            if not file_exists:
                writer.writerow(self.FEATURE_COLUMNS + ['direction', 'target', 'market_state', 'entry_setup', 'spread_at_entry'])
            writer.writerow([latest_row[c] for c in self.FEATURE_COLUMNS] + [direction, target, market_state, entry_setup, spread])

    def log_trade_journal(self, record):
        """Log trade to journal CSV"""
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
        """Get next trade ID"""
        if not os.path.exists(self.trade_journal_path):
            return 1
        df = pd.read_csv(self.trade_journal_path)
        if df.empty:
            return 1
        return int(df['Trade_ID'].max()) + 1

    def retrain_ai_from_memory(self):
        """Retrain AI model from trade history"""
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
        logging.info("🧠 AI Model retrained with %d trades", len(clean))

    @staticmethod
    def performance_metrics(df):
        """Calculate performance statistics"""
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
        return {
            'trades': trades,
            'wins': wins,
            'losses': losses,
            'win_rate_pct': round(win_rate, 2),
            'profit_factor': round(pf, 3) if np.isfinite(pf) else 'inf',
            'roi_pct': round(roi, 2),
            'max_drawdown_pct': round(mdd, 2)
        }

    def report_performance(self, source=None):
        """Report performance statistics"""
        if not os.path.exists(self.trade_journal_path):
            logging.warning("No trade journal found")
            return
        df = pd.read_csv(self.trade_journal_path)
        if source:
            df = df[df['source'] == source]
        if df.empty:
            logging.warning("No trades found for source=%s", source)
            return
        stats = self.performance_metrics(df)
        logging.info("=" * 60)
        logging.info("PERFORMANCE REPORT [%s]", source or 'ALL')
        logging.info("=" * 60)
        for key, value in stats.items():
            logging.info("%s: %s", key.replace('_', ' ').title(), value)
        logging.info("=" * 60)

    def run(self):
        """Main trading loop"""
        logging.info("Starting trading loop...")
        
        while True:
            self.reset_daily_trades()
            
            # Check daily limits
            if self.daily_trade_count >= self.max_trades_per_day:
                logging.info("Daily trade cap reached. Sleeping...")
                time.sleep(60 * 30)
                continue
            
            if self.daily_trade_count >= self.phase_trade_cap:
                logging.info("Phase trade cap reached (%d). Sleeping...", self.phase_trade_cap)
                time.sleep(60 * 30)
                continue
            
            # Check drawdown
            daily_dd = ((self.start_of_day_balance - self.balance) / self.start_of_day_balance) * 100
            if daily_dd >= self.max_daily_drawdown_pct:
                logging.warning("Daily drawdown limit hit (%.1f%%). Stopping for today.", daily_dd)
                time.sleep(60 * 60)
                continue
            
            # Check cooldown
            if self.last_trade_time:
                elapsed = datetime.datetime.now(datetime.timezone.utc) - self.last_trade_time
                if elapsed < datetime.timedelta(minutes=self.trade_cooldown_minutes):
                    time.sleep(30)
                    continue
            
            # Scan pairs
            for pair in self.pairs:
                if self.daily_trade_count >= self.max_trades_per_day:
                    break
                
                try:
                    # Get market data
                    df = self.get_market_data(pair, timeframe='1m', limit=240)
                    mtf = self.get_multi_timeframe_snapshot(pair, limit=100)
                    
                    if df is None or len(df) == 0:
                        continue
                    
                    # Get HTF bias
                    mtf_bias = self.timeframe_bias(mtf)
                    
                    # Analyze with HYBRID system
                    direction = self.analyze_ict_hybrid(df, pair=pair, mtf_bias=mtf_bias, log_signal=True)
                    
                    if not direction:
                        continue
                    
                    # Calculate trade parameters
                    entry = float(df['close'].iloc[-1])
                    atr = float(df['atr'].iloc[-1])
                    
                    if np.isnan(atr) or atr <= 0:
                        continue
                    
                    if self.is_crypto(pair):
                        stop = entry - (atr * 1.8) if direction == 'buy' else entry + (atr * 1.8)
                    else:
                        stop = entry - atr if direction == 'buy' else entry + atr
                    
                    risk_amount = self.balance * (self.risk_pct / 100)
                    risk_per_trade = max(abs(entry - stop), 0.0001)
                    size = abs(risk_amount / risk_per_trade)
                    
                    market_state = "Volatile"  # Could enhance this
                    entry_setup = 'ICT_HYBRID'
                    
                    # Execute trade
                    current_idx = len(df) - 1
                    self.execute_trade(
                        pair, direction, size, df.iloc[-1],
                        entry, stop, market_state, entry_setup,
                        df=df, current_idx=current_idx
                    )
                    
                    if self.daily_trade_count >= self.phase_trade_cap:
                        break
                
                except Exception as exc:
                    logging.exception("Error processing %s: %s", pair, exc)
                    continue
            
            # Sleep between scans
            time.sleep(120)

    def run_backtest(self, bars=800, payout=0.82):
        """Run backtest"""
        logging.info("=" * 60)
        logging.info("STARTING BACKTEST")
        logging.info("=" * 60)
        
        for pair in self.pairs:
            try:
                df = self.get_market_data(pair, timeframe='1m', limit=bars)
            except Exception as exc:
                logging.warning("Skipping %s: %s", pair, exc)
                continue
            
            equity = self.balance
            rows = []
            
            for i in range(100, len(df) - 6):
                window = df.iloc[:i + 1]
                mtf = self.get_multi_timeframe_snapshot(pair, limit=100)
                mtf_bias = self.timeframe_bias(mtf)
                
                direction = self.analyze_ict_hybrid(window, pair=pair, mtf_bias=mtf_bias, log_signal=False)
                if not direction:
                    continue
                
                entry = float(df['close'].iloc[i])
                risk_amount = equity * (self.risk_pct / 100)
                
                # Realistic outcome
                lookhead = 5
                future_prices = df['close'].iloc[i + 1:i + 1 + lookhead]
                
                if direction == 'buy':
                    won = future_prices.max() > entry
                else:
                    won = future_prices.min() < entry
                
                profit = risk_amount * payout if won else -risk_amount
                spread_cost = (1.5 / 10000) * risk_amount
                profit -= spread_cost
                
                equity += profit
                
                rows.append({
                    'Trade_ID': len(rows) + 1,
                    'timestamp': df['timestamp'].iloc[i + 1].isoformat(),
                    'pair': pair,
                    'direction': direction,
                    'entry': entry,
                    'stop': np.nan,
                    'size': risk_amount,
                    'profit': profit,
                    'balance_after': equity,
                    'source': 'backtest',
                    'won': 1 if won else 0,
                    'Market_State': 'Unknown',
                    'Time_of_Day': int(df['timestamp'].iloc[i + 1].hour),
                    'Entry_Setup': 'ICT_HYBRID',
                    'Spread_at_Entry': 1.5,
                    'Hold_Minutes': lookhead,
                    'Reward_Score': profit
                })
            
            for rec in rows:
                self.log_trade_journal(rec)
            
            if rows:
                stats = self.performance_metrics(pd.DataFrame(rows))
                logging.info("%s BACKTEST RESULTS: %s", pair, stats)


def parse_args():
    parser = argparse.ArgumentParser(description="Pocket Option ICT Bot - Hybrid Edition")
    parser.add_argument('--mode', choices=['run', 'backtest', 'report'], default='run')
    parser.add_argument('--bars', type=int, default=800)
    parser.add_argument('--payout', type=float, default=0.82)
    parser.add_argument('--source', choices=['all', 'demo', 'live', 'backtest'], default='all')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    bot = PocketOptionBot()
    
    if args.mode == 'backtest':
        bot.run_backtest(bars=args.bars, payout=args.payout)
    elif args.mode == 'report':
        bot.report_performance(source=None if args.source == 'all' else args.source)
    else:
        bot.run()