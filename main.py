"""
ICT TRADING BOT - KRAKEN EDITION
Combines ICT Unified Protocol + Scoring System + AI Learning
Uses Kraken for BOTH market data AND trade execution.

DEMO_MODE=True  → Kraken Futures Sandbox (demo-futures.kraken.com) - play money
DEMO_MODE=False → Kraken Futures Live (futures.kraken.com) - real money
"""
import warnings
warnings.filterwarnings(action="ignore", message="datetime.datetime.now(datetime.UTC)")
import argparse
import csv
import datetime
import json
import logging
import os
import pickle
import time

import ccxt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

import config

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[logging.StreamHandler()]
)


# ══════════════════════════════════════════════════════════════
# AI BRAIN
# ══════════════════════════════════════════════════════════════

class ProbabilityBrain:
    """Logistic-regression brain trained on historical + live trades."""

    def __init__(self):
        self.mean    = None
        self.std     = None
        self.weights = None
        self.bias    = 0.0

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y, l2=0.1):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean = X.mean(axis=0)
        self.std  = X.std(axis=0)
        self.std[self.std == 0] = 1.0
        Xn    = (X - self.mean) / self.std
        X_aug = np.c_[np.ones((Xn.shape[0], 1)), Xn]
        eye   = np.eye(X_aug.shape[1])
        eye[0, 0] = 0.0
        params       = np.linalg.solve(X_aug.T @ X_aug + l2 * eye, X_aug.T @ y)
        self.bias    = float(params[0])
        self.weights = params[1:]

    def predict_proba(self, X):
        X  = np.asarray(X, dtype=float)
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


# ══════════════════════════════════════════════════════════════
# MAIN BOT
# ══════════════════════════════════════════════════════════════

class ICTKrakenBot:
    FEATURE_COLUMNS    = ['rsi', 'ema_fast', 'ema_slow', 'atr', 'volume']
    BASE_OHLCV_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    # Pair → Kraken spot market symbol
    SPOT_MARKETS = {
        'EURUSD': 'EUR/USD',
        'GBPUSD': 'GBP/USD',
        'ETHUSD': 'ETH/USD',
        'BTCUSD': 'BTC/USD',
    }

    # Pair → Kraken Futures contract symbol
    FUTURES_MARKETS = {
        'EURUSD': 'EUR/USD:USD',
        'GBPUSD': 'GBP/USD:USD',
        'ETHUSD': 'ETH/USD:USD',
        'BTCUSD': 'BTC/USD:USD',
    }

    def __init__(self):
        self.demo_mode      = config.DEMO_MODE
        self.balance        = config.START_BALANCE
        self.start_of_day_balance = self.balance

        # Trading settings
        self.pairs                       = config.PAIRS
        self.timeframe                   = config.TIMEFRAME
        self.risk_pct                    = config.RISK_PERCENTAGE
        self.max_trades_per_day          = config.MAX_TRADES_PER_DAY
        self.max_daily_drawdown_pct      = config.MAX_DAILY_DRAWDOWN_PCT
        self.trade_cooldown_minutes      = config.TRADE_COOLDOWN_MINUTES
        self.max_account_use_per_trade_pct = config.MAX_ACCOUNT_USE_PER_TRADE_PCT
        self.training_phase              = config.TRAINING_PHASE

        # ICT settings
        self.enable_time_filter  = config.ENABLE_TIME_FILTER
        self.london_killzone     = config.LONDON_KILLZONE_UTC
        self.ny_killzone         = config.NY_KILLZONE_UTC
        self.liquidity_lookback  = config.LIQUIDITY_LOOKBACK
        self.sweep_memory        = config.SWEEP_MEMORY_CANDLES
        self.min_signal_score    = config.MIN_SIGNAL_SCORE

        # AI settings
        self.enable_ai_filter         = config.ENABLE_AI_FILTER
        self.ai_model_path            = config.AI_MODEL_PATH
        self.trade_memory_path        = config.TRADE_MEMORY_PATH
        self.trade_journal_path       = config.TRADE_JOURNAL_PATH
        self.ai_min_buy_prob          = config.AI_MIN_BUY_PROB
        self.ai_max_sell_prob         = config.AI_MAX_SELL_PROB
        self.ai_retrain_every_n_trades = config.AI_RETRAIN_EVERY_N_TRADES
        self.ai_min_retrain_rows      = config.AI_MIN_RETRAIN_ROWS

        # State
        self.daily_trade_count  = 0
        self.today              = datetime.date.today()
        self.last_trade_time    = None
        self.last_signal_profile = None
        self.model              = self._load_ai_model()
        self.signal_weights_path = 'signal_weights.json'
        self.signal_weights     = self._load_signal_weights()

        # ── Kraken exchanges ─────────────────────────────────────────────
        # Spot (for market data on forex/crypto)
        self.spot_exchange = ccxt.kraken({'enableRateLimit': True})

        # Futures (for actual trade execution)
        # Demo  → Kraken Futures Sandbox
        # Live  → Kraken Futures production
        futures_config = {
            'apiKey':        config.KRAKEN_API_KEY,
            'secret':        config.KRAKEN_API_SECRET,
            'enableRateLimit': True,
        }
        if self.demo_mode:
            futures_config['urls'] = {
                'api': {
                    'public':  'https://demo-futures.kraken.com/derivatives/api/v3',
                    'private': 'https://demo-futures.kraken.com/derivatives/api/v3',
                }
            }
            self.trading_exchange = ccxt.krakenfutures(futures_config)
        else:
            self.trading_exchange = ccxt.krakenfutures(futures_config)

        logging.info("=" * 60)
        logging.info("ICT KRAKEN BOT")
        logging.info("=" * 60)
        logging.info("Mode      : %s", "DEMO (Kraken Sandbox)" if self.demo_mode else "LIVE (Kraken Futures)")
        logging.info("Balance   : $%.2f", self.balance)
        logging.info("Pairs     : %s", ", ".join(self.pairs))
        logging.info("Phase     : %s (cap %d trades/day)", self.training_phase, self.phase_trade_cap)
        logging.info("=" * 60)

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def phase_trade_cap(self):
        if self.training_phase == 'month1':
            return 100
        if self.training_phase == 'month2':
            return 30
        return 3

    def is_crypto(self, pair):
        return pair in ('ETHUSD', 'BTCUSD')

    # ── AI ───────────────────────────────────────────────────────────────

    def _load_ai_model(self):
        if not self.enable_ai_filter:
            return None
        if os.path.exists(self.ai_model_path):
            m = ProbabilityBrain.load(self.ai_model_path)
            logging.info("Loaded AI model from %s", self.ai_model_path)
            return m
        logging.warning("AI model not found. Starting fresh.")
        return None

    def _ai_probability_up(self, latest_row):
        if self.model is None:
            return None
        vals = latest_row[self.FEATURE_COLUMNS]
        if vals.isna().any():
            return None
        return float(self.model.predict_proba(
            vals.values.astype(float).reshape(1, -1))[0][1])

    def retrain_ai_from_memory(self):
        if not os.path.exists(self.trade_memory_path):
            return
        mem = pd.read_csv(self.trade_memory_path)
        if len(mem) < self.ai_min_retrain_rows:
            return
        clean = mem.dropna(subset=self.FEATURE_COLUMNS + ['target'])
        if clean.empty:
            return
        X = clean[self.FEATURE_COLUMNS].values
        y = clean['target'].astype(int).values
        self.model = ProbabilityBrain()
        self.model.fit(X, y)
        self.model.save(self.ai_model_path)
        logging.info("🧠 AI retrained on %d trades", len(clean))

    # ── Signal weights ───────────────────────────────────────────────────

    def _load_signal_weights(self):
        defaults = {
            "liq_sweep_buy":      5.0,
            "liq_sweep_sell":     5.0,
            "displacement_up":    3.0,
            "displacement_down":  3.0,
            "structure_break_up": 3.0,
            "structure_break_down": 3.0,
            "fvg_up":   1.5,
            "fvg_down": 1.5,
            "mtf_bias": 2.0,
            "rsi_zone": 1.0,
        }
        if os.path.exists(self.signal_weights_path):
            try:
                with open(self.signal_weights_path) as fh:
                    saved = json.load(fh)
                for k, v in saved.items():
                    if k in defaults:
                        defaults[k] = float(v)
                logging.info("Loaded signal weights from %s", self.signal_weights_path)
            except Exception as exc:
                logging.warning("Could not load signal weights: %s", exc)
        return defaults

    def _save_signal_weights(self):
        try:
            with open(self.signal_weights_path, 'w') as fh:
                json.dump(self.signal_weights, fh, indent=2)
        except Exception as exc:
            logging.warning("Could not save signal weights: %s", exc)

    def _update_signal_weights(self, profile, won):
        if not profile:
            return
        step = 0.05 if won else -0.025
        for name, triggered in profile.items():
            if not triggered:
                continue
            cur = float(self.signal_weights.get(name, 1.0))
            self.signal_weights[name] = float(np.clip(cur + step, 0.5, 6.0))
        self._save_signal_weights()

    # ── Market data ──────────────────────────────────────────────────────

    def get_market_data(self, symbol, timeframe='1m', limit=240):
        """Fetch OHLCV from Kraken spot and enrich with indicators."""
        market = self.SPOT_MARKETS.get(symbol)
        if not market:
            logging.error("Unknown pair: %s", symbol)
            return None

        for attempt in range(3):
            try:
                candles = self.spot_exchange.fetch_ohlcv(
                    market, timeframe=timeframe, limit=limit)
                df = pd.DataFrame(candles, columns=self.BASE_OHLCV_COLUMNS)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return self._enrich(df)
            except (ccxt.RequestTimeout, ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
                wait = (attempt + 1) * 5
                logging.warning("Kraken timeout attempt %d/3. Retry in %ds: %s", attempt + 1, wait, e)
                time.sleep(wait)
            except Exception as e:
                logging.error("get_market_data error for %s: %s", symbol, e)
                return None
        logging.error("Kraken unresponsive after 3 attempts. Skipping %s.", symbol)
        return None

    def _enrich(self, df):
        """Add technical indicators."""
        out = df.copy()
        out['ema_fast'] = out['close'].ewm(span=20, adjust=False).mean()
        out['ema_slow'] = out['close'].ewm(span=50, adjust=False).mean()

        delta = out['close'].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        out['rsi'] = 100 - (100 / (1 + rs))

        tr = pd.concat([
            out['high'] - out['low'],
            (out['high'] - out['close'].shift(1)).abs(),
            (out['low']  - out['close'].shift(1)).abs(),
        ], axis=1)
        out['atr']    = tr.max(axis=1).rolling(14).mean()
        out['vol_ma'] = out['volume'].rolling(20).mean()
        return out

    # Alias for train_ai.py compatibility
    def enrich_market_data(self, df):
        return self._enrich(df)

    def get_multi_timeframe_snapshot(self, symbol, limit=100):
        snapshots = {}
        for tf in config.HIGHER_TIMEFRAMES:
            snapshots[tf] = self.get_market_data(symbol, timeframe=tf, limit=limit)
        return snapshots

    def timeframe_bias(self, snapshot):
        up = down = 0
        for df in snapshot.values():
            if df is None or df.empty:
                continue
            if df['ema_fast'].iloc[-1] > df['ema_slow'].iloc[-1]:
                up += 1
            elif df['ema_fast'].iloc[-1] < df['ema_slow'].iloc[-1]:
                down += 1
        if up > down:   return 'buy'
        if down > up:   return 'sell'
        return None

    # ── ICT Analysis ─────────────────────────────────────────────────────

    def analyze_ict_hybrid(self, df, pair=None, mtf_bias=None, log_signal=True):
        """
        HYBRID ICT ANALYSIS
        Gate 1 → Time filter (Silver Bullet killzones)
        Gate 2 → Liquidity sweep
        Gate 3 → Structure break + displacement
        Score  → Weighted signal scoring
        AI     → Final probability filter
        """
        if df is None or len(df) < 50:
            return None

        required = {'open', 'high', 'low', 'close', 'ema_fast', 'ema_slow', 'rsi', 'atr', 'timestamp'}
        if not required.issubset(df.columns):
            return None

        work       = df.copy()
        latest     = work.iloc[-1]
        hour_utc   = pd.to_datetime(latest['timestamp']).hour

        # ── Gate 1: Killzone ────────────────────────────────────────────
        if self.enable_time_filter:
            in_london = self.london_killzone[0] <= hour_utc < self.london_killzone[1]
            in_ny     = self.ny_killzone[0]     <= hour_utc < self.ny_killzone[1]
            if not (in_london or in_ny):
                return None

        # ── Compute ICT signals ─────────────────────────────────────────
        prev_high = work['high'].rolling(self.liquidity_lookback).max().shift(1)
        prev_low  = work['low'].rolling(self.liquidity_lookback).min().shift(1)

        work['liq_sweep_buy']  = (work['low']  < prev_low)  & (work['close'] > prev_low)
        work['liq_sweep_sell'] = (work['high'] > prev_high) & (work['close'] < prev_high)

        work['body']           = (work['close'] - work['open']).abs()
        work['displacement']   = work['body'] > (work['atr'] * 0.8)

        work['structure_break_up']   = work['close'] > work['high'].shift(5).rolling(5).max()
        work['structure_break_down'] = work['close'] < work['low'].shift(5).rolling(5).min()

        work['fvg_up']   = work['low']  > work['high'].shift(2)
        work['fvg_down'] = work['high'] < work['low'].shift(2)

        latest = work.iloc[-1]

        # ── Gate 2 & 3 ─────────────────────────────────────────────────
        recent_sweep_buy  = any(work['liq_sweep_buy'].tail(self.sweep_memory))
        recent_sweep_sell = any(work['liq_sweep_sell'].tail(self.sweep_memory))

        disp_up   = bool(latest['displacement']) and latest['close'] > latest['open']
        disp_down = bool(latest['displacement']) and latest['close'] < latest['open']
        str_up    = bool(latest['structure_break_up'])
        str_down  = bool(latest['structure_break_down'])

        bullish = recent_sweep_buy  and disp_up   and str_up
        bearish = recent_sweep_sell and disp_down and str_down

        if not (bullish or bearish):
            return None

        # ── Scoring ─────────────────────────────────────────────────────
        is_crypto   = self.is_crypto(pair) if pair else False
        rsi_buy_zone  = (50 <= latest['rsi'] <= 70) if is_crypto else (45 <= latest['rsi'] <= 65)
        rsi_sell_zone = (30 <= latest['rsi'] <= 50) if is_crypto else (35 <= latest['rsi'] <= 55)

        signals = {
            "liq_sweep_buy":      bullish,
            "liq_sweep_sell":     bearish,
            "displacement_up":    disp_up,
            "displacement_down":  disp_down,
            "structure_break_up": str_up,
            "structure_break_down": str_down,
            "fvg_up":   bool(any(work['fvg_up'].tail(3))),
            "fvg_down": bool(any(work['fvg_down'].tail(3))),
            "mtf_bias": mtf_bias is not None,
            "rsi_zone": False,
        }

        buy_score = sell_score = 0.0
        w = self.signal_weights

        if signals['liq_sweep_buy']:      buy_score  += w['liq_sweep_buy']
        if signals['liq_sweep_sell']:     sell_score += w['liq_sweep_sell']
        if signals['displacement_up']:    buy_score  += w['displacement_up']
        if signals['displacement_down']:  sell_score += w['displacement_down']
        if signals['structure_break_up']: buy_score  += w['structure_break_up']
        if signals['structure_break_down']: sell_score += w['structure_break_down']
        if signals['fvg_up']:             buy_score  += w['fvg_up']
        if signals['fvg_down']:           sell_score += w['fvg_down']

        if mtf_bias == 'buy':
            buy_score  += w['mtf_bias']
            sell_score  = 0
        elif mtf_bias == 'sell':
            sell_score += w['mtf_bias']
            buy_score   = 0

        if rsi_buy_zone:
            buy_score += w['rsi_zone']
            signals['rsi_zone'] = True
        if rsi_sell_zone:
            sell_score += w['rsi_zone']
            signals['rsi_zone'] = True

        direction = None
        if buy_score  >= self.min_signal_score and buy_score  > sell_score:
            direction = 'buy'
        elif sell_score >= self.min_signal_score and sell_score > buy_score:
            direction = 'sell'

        if direction is None:
            return None

        # ── AI filter ───────────────────────────────────────────────────
        prob_up = self._ai_probability_up(latest)
        if prob_up is not None:
            if direction == 'buy'  and prob_up < self.ai_min_buy_prob:
                return None
            if direction == 'sell' and prob_up > self.ai_max_sell_prob:
                return None

        self.last_signal_profile = signals

        if log_signal:
            score = buy_score if direction == 'buy' else sell_score
            logging.info(
                "✅ SIGNAL: %s | Score: %.1f | MTF: %s | AI: %.2f | %02d:00 UTC",
                direction.upper(), score,
                mtf_bias or 'neutral',
                prob_up if prob_up else 0.5,
                hour_utc,
            )
        return direction

    # ── Trade execution ──────────────────────────────────────────────────

    def _get_live_balance(self):
        """Fetch real account balance from Kraken."""
        try:
            bal = self.trading_exchange.fetch_balance()
            # Kraken Futures returns total in USD
            usd = bal.get('total', {}).get('USD', None)
            if usd is not None:
                return float(usd)
        except Exception as exc:
            logging.warning("Could not fetch live balance: %s", exc)
        return self.balance  # fallback to tracked balance

    def execute_trade(self, pair, direction, size, latest_row,
                      entry, stop, market_state, entry_setup,
                      df=None, current_idx=None):
        """Place trade on Kraken Futures (demo or live)."""
        self.daily_trade_count += 1
        stake = min(size, self.balance * (self.max_account_use_per_trade_pct / 100))
        stake = max(stake, 1.0)  # Kraken minimum

        spread_pips = (
            np.random.uniform(2.0, 5.0) if self.is_crypto(pair)
            else np.random.uniform(1.5, 2.5)
        )

        # ── Execute on Kraken ────────────────────────────────────────────
        profit      = 0.0
        hold_minutes = 5
        won_flag    = 0

        try:
            market  = self.FUTURES_MARKETS.get(pair)
            side    = 'buy' if direction == 'buy' else 'sell'
            # Size in contracts (1 contract ≈ $1 for most Kraken linear futures)
            contracts = round(stake, 2)

            order = self.trading_exchange.create_market_order(
                symbol=market,
                side=side,
                amount=contracts,
            )
            order_id = order.get('id', 'unknown')
            logging.info("📤 Order placed: %s %s %.2f contracts | ID: %s",
                         side.upper(), pair, contracts, order_id)

            # Wait for the trade window then close
            time.sleep(config.TRADE_HOLD_SECONDS)

            # Close the position
            close_side = 'sell' if direction == 'buy' else 'buy'
            close_order = self.trading_exchange.create_market_order(
                symbol=market,
                side=close_side,
                amount=contracts,
                params={'reduceOnly': True},
            )

            # Fetch updated balance to calculate P&L
            new_balance = self._get_live_balance()
            profit      = new_balance - self.balance
            won_flag    = 1 if profit > 0 else 0
            hold_minutes = config.TRADE_HOLD_SECONDS // 60

        except ccxt.AuthenticationError:
            logging.error("❌ Kraken auth failed — check KRAKEN_API_KEY / KRAKEN_API_SECRET in Railway Variables.")
            self.daily_trade_count -= 1
            return
        except ccxt.InsufficientFunds:
            logging.error("❌ Insufficient funds on Kraken account.")
            self.daily_trade_count -= 1
            return
        except Exception as exc:
            logging.error("❌ Trade execution error: %s", exc)
            self.daily_trade_count -= 1
            return

        # ── Update state ────────────────────────────────────────────────
        old_balance  = self.balance
        self.balance += profit
        self.last_trade_time = datetime.datetime.now(datetime.timezone.utc)

        profit_pct = (profit / max(stake, 0.01)) * 100
        reward     = profit_pct - (spread_pips * 0.5)

        self._log_trade_memory(latest_row, direction, profit, entry_setup, market_state, spread_pips)
        self._log_trade_journal({
            'Trade_ID':       self._next_trade_id(),
            'timestamp':      self.last_trade_time.isoformat(),
            'pair':           pair,
            'direction':      direction,
            'entry':          round(float(entry), 8),
            'stop':           round(float(stop), 8),
            'size':           round(float(stake), 2),
            'profit':         round(float(profit), 2),
            'balance_after':  round(float(self.balance), 2),
            'source':         'demo' if self.demo_mode else 'live',
            'won':            won_flag,
            'Market_State':   market_state,
            'Time_of_Day':    self.last_trade_time.hour,
            'Entry_Setup':    entry_setup,
            'Spread_at_Entry': round(float(spread_pips), 2),
            'Hold_Minutes':   int(hold_minutes),
            'Reward_Score':   round(float(reward), 2),
        })

        if self.ai_retrain_every_n_trades > 0 and \
           self.daily_trade_count % self.ai_retrain_every_n_trades == 0:
            self.retrain_ai_from_memory()

        self._update_signal_weights(self.last_signal_profile, profit > 0)

        label = "WIN ✅" if won_flag else "LOSS ❌"
        pnl   = f"+${profit:.2f}" if profit > 0 else f"-${abs(profit):.2f}"
        logging.info(
            "🎯 TRADE #%d: %s %s | Stake: $%.2f | %s %s | Balance: $%.2f → $%.2f (Daily: %+.2f%%)",
            self.daily_trade_count, pair, direction.upper(), stake,
            label, pnl, old_balance, self.balance,
            ((self.balance - self.start_of_day_balance) / self.start_of_day_balance * 100),
        )

    # ── Logging helpers ──────────────────────────────────────────────────

    def _log_trade_memory(self, latest_row, direction, profit, entry_setup, market_state, spread):
        target      = 1 if profit > 0 else 0
        file_exists = os.path.isfile(self.trade_memory_path)
        with open(self.trade_memory_path, 'a', newline='') as fh:
            w = csv.writer(fh)
            if not file_exists:
                w.writerow(self.FEATURE_COLUMNS +
                           ['direction', 'target', 'market_state', 'entry_setup', 'spread_at_entry'])
            w.writerow([latest_row[c] for c in self.FEATURE_COLUMNS] +
                       [direction, target, market_state, entry_setup, spread])

    def _log_trade_journal(self, record):
        fields = [
            'Trade_ID', 'timestamp', 'pair', 'direction', 'entry', 'stop',
            'size', 'profit', 'balance_after', 'source', 'won',
            'Market_State', 'Time_of_Day', 'Entry_Setup',
            'Spread_at_Entry', 'Hold_Minutes', 'Reward_Score',
        ]
        file_exists = os.path.isfile(self.trade_journal_path)
        with open(self.trade_journal_path, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            if not file_exists:
                w.writeheader()
            w.writerow(record)

    def _next_trade_id(self):
        if not os.path.exists(self.trade_journal_path):
            return 1
        try:
            df = pd.read_csv(self.trade_journal_path)
            return 1 if df.empty else int(df['Trade_ID'].max()) + 1
        except Exception:
            return 1

    # ── Daily reset ──────────────────────────────────────────────────────

    def reset_daily_trades(self):
        if datetime.date.today() != self.today:
            logging.info("=" * 60)
            logging.info("NEW DAY - Resetting counters")
            logging.info("Yesterday P/L: $%.2f", self.balance - self.start_of_day_balance)
            logging.info("=" * 60)
            self.daily_trade_count    = 0
            self.today                = datetime.date.today()
            self.start_of_day_balance = self.balance
            self.last_trade_time      = None

    # ── Performance ──────────────────────────────────────────────────────

    @staticmethod
    def performance_metrics(df):
        trades = len(df)
        wins   = int((df['won'] == 1).sum())
        win_rate = (wins / trades * 100) if trades else 0.0
        gp = float(df.loc[df['profit'] > 0, 'profit'].sum())
        gl = float(-df.loc[df['profit'] < 0, 'profit'].sum())
        pf = (gp / gl) if gl > 0 else float('inf')
        sb = float(df['balance_after'].iloc[0] - df['profit'].iloc[0]) if trades else 0.0
        eb = float(df['balance_after'].iloc[-1]) if trades else 0.0
        roi = ((eb - sb) / sb * 100) if sb > 0 else 0.0
        eq  = df['balance_after'].astype(float)
        mdd = float(((eq - eq.cummax()) / eq.cummax()).min() * 100) if trades else 0.0
        return {
            'trades':           trades,
            'wins':             wins,
            'losses':           trades - wins,
            'win_rate_pct':     round(win_rate, 2),
            'profit_factor':    round(pf, 3) if np.isfinite(pf) else 'inf',
            'roi_pct':          round(roi, 2),
            'max_drawdown_pct': round(mdd, 2),
        }

    def report_performance(self, source=None):
        if not os.path.exists(self.trade_journal_path):
            logging.warning("No trade journal found")
            return
        df = pd.read_csv(self.trade_journal_path)
        if source:
            df = df[df['source'] == source]
        if df.empty:
            logging.warning("No trades for source=%s", source)
            return
        stats = self.performance_metrics(df)
        logging.info("=" * 60)
        logging.info("PERFORMANCE REPORT [%s]", source or 'ALL')
        logging.info("=" * 60)
        for k, v in stats.items():
            logging.info("%s: %s", k.replace('_', ' ').title(), v)
        logging.info("=" * 60)

    # ── Main loop ────────────────────────────────────────────────────────

    def run(self):
        logging.info("Starting trading loop...")

        while True:
            self.reset_daily_trades()

            if self.daily_trade_count >= min(self.max_trades_per_day, self.phase_trade_cap):
                logging.info("Trade cap reached. Sleeping 30 min...")
                time.sleep(60 * 30)
                continue

            daily_dd = ((self.start_of_day_balance - self.balance) /
                        self.start_of_day_balance) * 100
            if daily_dd >= self.max_daily_drawdown_pct:
                logging.warning("Daily drawdown limit %.1f%% hit. Sleeping 1h.", daily_dd)
                time.sleep(60 * 60)
                continue

            if self.last_trade_time:
                elapsed = datetime.datetime.now(datetime.timezone.utc) - self.last_trade_time
                if elapsed < datetime.timedelta(minutes=self.trade_cooldown_minutes):
                    time.sleep(30)
                    continue

            for pair in self.pairs:
                if self.daily_trade_count >= min(self.max_trades_per_day, self.phase_trade_cap):
                    break
                try:
                    df  = self.get_market_data(pair, timeframe='1m', limit=240)
                    mtf = self.get_multi_timeframe_snapshot(pair, limit=100)

                    if df is None or df.empty:
                        logging.warning("No data for %s, skipping.", pair)
                        continue

                    mtf_bias  = self.timeframe_bias(mtf)
                    direction = self.analyze_ict_hybrid(
                        df, pair=pair, mtf_bias=mtf_bias, log_signal=True)

                    if not direction:
                        continue

                    entry = float(df['close'].iloc[-1])
                    atr   = float(df['atr'].iloc[-1])
                    if np.isnan(atr) or atr <= 0:
                        continue

                    stop = (entry - atr * (1.8 if self.is_crypto(pair) else 1.0)
                            if direction == 'buy'
                            else entry + atr * (1.8 if self.is_crypto(pair) else 1.0))

                    risk_amount    = self.balance * (self.risk_pct / 100)
                    risk_per_trade = max(abs(entry - stop), 0.0001)
                    size           = abs(risk_amount / risk_per_trade)

                    self.execute_trade(
                        pair, direction, size, df.iloc[-1],
                        entry, stop,
                        market_state='Volatile',
                        entry_setup='ICT_HYBRID',
                        df=df, current_idx=len(df) - 1,
                    )

                except Exception as exc:
                    logging.exception("Error processing %s: %s", pair, exc)
                    continue

            time.sleep(120)

    # ── Backtest ─────────────────────────────────────────────────────────

    def run_backtest(self, bars=800, payout=0.82):
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
            rows   = []

            for i in range(100, len(df) - 6):
                window   = df.iloc[:i + 1]
                mtf      = self.get_multi_timeframe_snapshot(pair, limit=100)
                mtf_bias = self.timeframe_bias(mtf)
                direction = self.analyze_ict_hybrid(
                    window, pair=pair, mtf_bias=mtf_bias, log_signal=False)
                if not direction:
                    continue

                entry        = float(df['close'].iloc[i])
                risk_amount  = equity * (self.risk_pct / 100)
                future_slice = df['close'].iloc[i + 1:i + 6]

                won    = (future_slice.max() > entry if direction == 'buy'
                          else future_slice.min() < entry)
                profit = risk_amount * payout if won else -risk_amount
                profit -= (1.5 / 10000) * risk_amount
                equity += profit

                rows.append({
                    'Trade_ID':       len(rows) + 1,
                    'timestamp':      df['timestamp'].iloc[i + 1].isoformat(),
                    'pair':           pair,
                    'direction':      direction,
                    'entry':          entry,
                    'stop':           np.nan,
                    'size':           risk_amount,
                    'profit':         profit,
                    'balance_after':  equity,
                    'source':         'backtest',
                    'won':            1 if won else 0,
                    'Market_State':   'Unknown',
                    'Time_of_Day':    int(df['timestamp'].iloc[i + 1].hour),
                    'Entry_Setup':    'ICT_HYBRID',
                    'Spread_at_Entry': 1.5,
                    'Hold_Minutes':   5,
                    'Reward_Score':   profit,
                })

            for rec in rows:
                self._log_trade_journal(rec)

            if rows:
                stats = self.performance_metrics(pd.DataFrame(rows))
                logging.info("%s BACKTEST: %s", pair, stats)


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='ICT Kraken Bot')
    p.add_argument('--mode',   choices=['run', 'backtest', 'report'], default='run')
    p.add_argument('--bars',   type=int,   default=800)
    p.add_argument('--payout', type=float, default=0.82)
    p.add_argument('--source', choices=['all', 'demo', 'live', 'backtest'], default='all')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    bot  = ICTKrakenBot()

    if args.mode == 'backtest':
        bot.run_backtest(bars=args.bars, payout=args.payout)
    elif args.mode == 'report':
        bot.report_performance(source=None if args.source == 'all' else args.source)
    else:
        bot.run()