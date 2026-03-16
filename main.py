import warnings
# Ignore the specific DeprecationWarning about utcnow
warnings.filterwarnings(action="ignore", message="datetime.datetime.now(datetime.UTC)")
import argparse
import csv
import datetime
import logging
import os
import pickle
import time
import logging
import requests
import warnings

import ccxt
from ccxt.base.errors import NetworkError as CCXTNetworkError
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from websocket import create_connection

import config

warnings.filterwarnings(action="ignore", message="datetime.datetime.now(datetime.timezone.utc)")
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[logging.StreamHandler()]
    handlers=[logging.StreamHandler()],
)


class ProbabilityBrain:
    """Lightweight probability model for up/down prediction."""

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
        linear = Xn @ self.weights + self.bias
        p1 = self._sigmoid(linear)
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

    def __init__(self):
        self.email = config.PO_EMAIL
        self.password = config.PO_PASSWORD
        self.demo_mode = config.DEMO_MODE
        self.risk_pct = config.RISK_PERCENTAGE
        self.api_wss = config.PO_API_WSS
        self.max_trades = config.MAX_TRADES_PER_DAY
        self.min_signal_score = config.MIN_SIGNAL_SCORE
        self.max_daily_drawdown_pct = config.MAX_DAILY_DRAWDOWN_PCT
        self.trade_cooldown_minutes = config.TRADE_COOLDOWN_MINUTES
        self.market_data_limit = config.MARKET_DATA_LIMIT

        self.enable_ai_filter = config.ENABLE_AI_FILTER
        self.ai_model_path = config.AI_MODEL_PATH
        self.trade_memory_path = config.TRADE_MEMORY_PATH
        self.trade_journal_path = config.TRADE_JOURNAL_PATH
        self.ai_min_buy_prob = config.AI_MIN_BUY_PROB
        self.ai_max_sell_prob = config.AI_MAX_SELL_PROB
        self.ai_retrain_every_n_trades = config.AI_RETRAIN_EVERY_N_TRADES
        self.ai_min_retrain_rows = config.AI_MIN_RETRAIN_ROWS

        self.pairs = config.PAIRS
        self.timeframe = config.TIMEFRAME
        self.balance = config.START_BALANCE
        self.start_of_day_balance = self.balance

        self.daily_trade_count = 0
        self.today = datetime.date.today()
        self.ws = None  # Will hold the websocket connection
        self.last_trade_time = None

        self.model = self.load_ai_model()

    @staticmethod
    def symbol_to_market(symbol):
        mapping = {'EURUSD': 'EUR/USDT', 'GBPUSD': 'GBP/USDT'}
        if symbol not in mapping:
            raise ValueError(f"Unknown pair for demo data: {symbol}")
        return mapping[symbol]

    def load_ai_model(self):
        if not self.enable_ai_filter:
            logging.info("AI filter disabled by config.")
            return None
        if os.path.exists(self.ai_model_path):
            model = ProbabilityBrain.load(self.ai_model_path)
            logging.info("AI model loaded: %s", self.ai_model_path)
            return model
        logging.warning("AI model not found at %s. Using rule-based mode only.", self.ai_model_path)
        return None

    def reset_daily_trades(self):
        if datetime.date.today() != self.today:
            self.daily_trade_count = 0
            self.today = datetime.date.today()
            self.start_of_day_balance = self.balance
            self.last_trade_time = None
            logging.info("New trading day detected. Counters reset.")

    def fetch_account_balance(self):
        # In real application, fetch via Pocket Option API (or WebSocket/HTTP endpoint if available)
        # For now, DEMO ONLY!
        logging.info(f"Simulated balance check: ${self.balance:.2f}")
        logging.info("Simulated balance check: $%.2f", self.balance)
        return self.balance

    def risk_size(self, entry, stop):
        risk_amount = self.fetch_account_balance() * (self.risk_pct / 100)
        risk_per_trade = max(abs(entry - stop), 0.0001)
        size = abs(risk_amount / risk_per_trade)
        return round(size, 6)

    def get_market_data(self, symbol, limit=120):
        # Using Binance for free demo candles (substitute with real PO API for production)
        import ccxt
    def is_daily_drawdown_breached(self):
        drawdown_pct = ((self.start_of_day_balance - self.balance) / self.start_of_day_balance) * 100
        if drawdown_pct >= self.max_daily_drawdown_pct:
            logging.warning(
                "Daily drawdown limit hit: %.2f%% (limit %.2f%%). Trading paused for today.",
                drawdown_pct,
                self.max_daily_drawdown_pct,
            )
            return True
        return False

    def is_cooldown_over(self):
        if self.last_trade_time is None:
            return True
        elapsed = datetime.datetime.now(datetime.timezone.utc) - self.last_trade_time
        return elapsed >= datetime.timedelta(minutes=self.trade_cooldown_minutes)

    def get_market_data(self, symbol, limit=240):
        exchange = ccxt.binance()
        if symbol == 'EURUSD':
            market = 'EUR/USDT'
        elif symbol == 'GBPUSD':
            market = 'GBP/USDT'
        else:
            raise Exception("Unknown pair for demo data")
        df = exchange.fetch_ohlcv(market, timeframe='1m', limit=limit)
        df = pd.DataFrame(df, columns=['timestamp','open','high','low','close','volume'])
        market = self.symbol_to_market(symbol)
        try:
            candles = exchange.fetch_ohlcv(market, timeframe='1m', limit=limit)
        except CCXTNetworkError as exc:
            raise RuntimeError(f"Unable to fetch market data for {symbol}: {exc}") from exc

        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
        return self.enrich_market_data(df)

    def enrich_market_data(self, df):
        out = df.copy()
        out['ema_fast'] = out['close'].ewm(span=20, adjust=False).mean()
        out['ema_slow'] = out['close'].ewm(span=50, adjust=False).mean()

    def analyze_ict(self, df):
        # ----- MINIMAL ICT/SMART MONEY LOGIC (all-in-one for space) -----
        # 1. Market Structure: find swing highs/lows
        df['s_high'] = (df['high'].shift(2) < df['high'].shift(1)) & (df['high'].shift(1) > df['high'])  # peak
        df['s_low'] = (df['low'].shift(2) > df['low'].shift(1)) & (df['low'].shift(1) < df['low'])
        delta = out['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        out['rsi'] = 100 - (100 / (1 + rs))

        # 2. FVG scan: identify fair value gaps
        df['fvg_up'] = (df['low'] > df['high'].shift(2))
        df['fvg_down'] = (df['high'] < df['low'].shift(2))
        tr_components = pd.concat(
            [
                out['high'] - out['low'],
                (out['high'] - out['close'].shift(1)).abs(),
                (out['low'] - out['close'].shift(1)).abs(),
            ],
            axis=1,
        )
        out['atr'] = tr_components.max(axis=1).rolling(14).mean()
        out['vol_ma'] = out['volume'].rolling(20).mean()
        return out

        # 3. Session check (basic): only trade London/NY hours UTC
        now = datetime.datetime.now(datetime.UTC).hour
        session_ok = (7 <= now <= 17)
    def _ai_probability_up(self, latest_row):
        if self.model is None:
            return None

        # 4. Signal generation:
        bullish = any(df['s_low'][-5:]) and any(df['fvg_up'][-3:]) and session_ok
        bearish = any(df['s_high'][-5:]) and any(df['fvg_down'][-3:]) and session_ok
        features = latest_row[self.FEATURE_COLUMNS]
        if features.isna().any():
            return None
        values = features.values.astype(float).reshape(1, -1)
        return float(self.model.predict_proba(values)[0][1])

        signal = None
        if bullish and not bearish:
            signal = 'buy'
        elif bearish and not bullish:
            signal = 'sell'
        return signal
    def analyze_ict(self, df, session_hour=None, log_signal=True):
        work = df.copy()
        work['s_high'] = (work['high'].shift(2) < work['high'].shift(1)) & (work['high'].shift(1) > work['high'])
        work['s_low'] = (work['low'].shift(2) > work['low'].shift(1)) & (work['low'].shift(1) < work['low'])
        work['fvg_up'] = work['low'] > work['high'].shift(2)
        work['fvg_down'] = work['high'] < work['low'].shift(2)

        if session_hour is None:
            session_hour = datetime.datetime.now(datetime.timezone.utc).hour
        if not (7 <= int(session_hour) <= 17):
            return None

        latest = work.iloc[-1]
        bullish_score = 0
        bearish_score = 0

        if any(work['s_low'].tail(5)):
            bullish_score += 2
        if any(work['fvg_up'].tail(3)):
            bullish_score += 2
        if latest['ema_fast'] > latest['ema_slow']:
            bullish_score += 2
        if 45 <= latest['rsi'] <= 65:
            bullish_score += 1
        if latest['volume'] > latest['vol_ma']:
            bullish_score += 1

        if any(work['s_high'].tail(5)):
            bearish_score += 2
        if any(work['fvg_down'].tail(3)):
            bearish_score += 2
        if latest['ema_fast'] < latest['ema_slow']:
            bearish_score += 2
        if 35 <= latest['rsi'] <= 55:
            bearish_score += 1
        if latest['volume'] > latest['vol_ma']:
            bearish_score += 1

        direction = None
        confidence = 0
        if bullish_score >= self.min_signal_score and bullish_score > bearish_score:
            direction = 'buy'
            confidence = bullish_score
        elif bearish_score >= self.min_signal_score and bearish_score > bullish_score:
            direction = 'sell'
            confidence = bearish_score

        if direction is None:
            return None

        prob_up = self._ai_probability_up(latest)
        if prob_up is not None:
            if direction == 'buy' and prob_up < self.ai_min_buy_prob:
                if log_signal:
                    logging.info("AI veto BUY: p_up=%.2f below %.2f", prob_up, self.ai_min_buy_prob)
                return None
            if direction == 'sell' and prob_up > self.ai_max_sell_prob:
                if log_signal:
                    logging.info("AI veto SELL: p_up=%.2f above %.2f", prob_up, self.ai_max_sell_prob)
                return None

        if log_signal:
            ai_text = f"ai_p_up={prob_up:.2f}" if prob_up is not None else "ai=off"
            logging.info(
                "%s signal | confidence=%s | bull=%s bear=%s | %s",
                direction.upper(),
                confidence,
                bullish_score,
                bearish_score,
                ai_text,
            )
        return direction

    def ensure_api_freshness(self):
        # Dummy implementation for demonstration. Real trading bots fetch fresh endpoint config from PO docs or via API gateway
        # This could be honed to auto-discover healthiest endpoint.
        logging.info("Checked Pocket Option API endpoint freshness (simulated auto-discovery)")

    def trade(self, pair, direction, size):
        # Simulate placing a trade via API.
        self.fetch_account_balance()  # refresh
        if self.demo_mode:
            # "Place" the trade and simulate instant result (for demo)
            self.daily_trade_count += 1
            change = np.random.uniform(0.75, 1.15) if direction == 'buy' else np.random.uniform(0.75, 1.15) * -1
            profit = size * change
            self.balance += profit
            outcome = "Profit" if profit > 0 else "Loss"
            logging.info(f"TRADE: {pair} | {direction.upper()} | Size {size} | {outcome} ${abs(profit):.2f} | New Balance: ${self.balance:.2f}")
        else:
            # Here you integrate a real API call to Pocket Option's live websocket (not demo)
    def log_trade_memory(self, latest_row, direction, profit):
        target = 1 if profit > 0 else 0
        file_exists = os.path.isfile(self.trade_memory_path)

        with open(self.trade_memory_path, mode='a', newline='') as fh:
            writer = csv.writer(fh)
            if not file_exists:
                writer.writerow(self.FEATURE_COLUMNS + ['direction', 'target'])
            writer.writerow([latest_row[c] for c in self.FEATURE_COLUMNS] + [direction, target])

    def log_trade_journal(self, record):
        file_exists = os.path.isfile(self.trade_journal_path)
        with open(self.trade_journal_path, mode='a', newline='') as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    'timestamp', 'pair', 'direction', 'entry', 'stop', 'size',
                    'profit', 'balance_after', 'source', 'won'
                ],
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)

    def retrain_ai_from_memory(self):
        if not os.path.exists(self.trade_memory_path):
            return

        memory_df = pd.read_csv(self.trade_memory_path)
        if len(memory_df) < self.ai_min_retrain_rows:
            logging.info("AI retrain skipped: %s rows < %s required.", len(memory_df), self.ai_min_retrain_rows)
            return

        clean = memory_df.dropna(subset=self.FEATURE_COLUMNS + ['target'])
        if clean.empty:
            logging.info("AI retrain skipped: no clean rows.")
            return

        X = clean[self.FEATURE_COLUMNS].values
        y = clean['target'].astype(int).values
        self.model = ProbabilityBrain()
        self.model.fit(X, y)
        self.model.save(self.ai_model_path)
        logging.info("AI model retrained from memory and saved to %s", self.ai_model_path)

    @staticmethod
    def performance_metrics(df):
        trades = len(df)
        wins = int((df['won'] == 1).sum())
        losses = trades - wins
        win_rate = (wins / trades * 100) if trades else 0.0

        gross_profit = float(df.loc[df['profit'] > 0, 'profit'].sum())
        gross_loss = float(-df.loc[df['profit'] < 0, 'profit'].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        start_balance = float(df['balance_after'].iloc[0] - df['profit'].iloc[0]) if trades else 0.0
        end_balance = float(df['balance_after'].iloc[-1]) if trades else 0.0
        roi_pct = ((end_balance - start_balance) / start_balance * 100) if start_balance > 0 else 0.0

        equity = df['balance_after'].astype(float)
        rolling_peak = equity.cummax()
        drawdown = (equity - rolling_peak) / rolling_peak
        max_drawdown_pct = float(drawdown.min() * 100) if trades else 0.0

        return {
            'trades': trades,
            'wins': wins,
            'losses': losses,
            'win_rate_pct': round(win_rate, 2),
            'profit_factor': round(profit_factor, 3) if np.isfinite(profit_factor) else 'inf',
            'roi_pct': round(roi_pct, 2),
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'start_balance': round(start_balance, 2),
            'end_balance': round(end_balance, 2),
        }

    def report_performance(self, source=None):
        if not os.path.exists(self.trade_journal_path):
            logging.warning("No trade journal found at %s", self.trade_journal_path)
            return
        df = pd.read_csv(self.trade_journal_path)
        if source:
            df = df[df['source'] == source]
        if df.empty:
            logging.warning("No rows in trade journal for source=%s", source)
            return

        stats = self.performance_metrics(df)
        logging.info(
            "PERFORMANCE [%s] | trades=%s wins=%s losses=%s win_rate=%s%% profit_factor=%s roi=%s%% max_dd=%s%% start=$%s end=$%s",
            source or 'all',
            stats['trades'],
            stats['wins'],
            stats['losses'],
            stats['win_rate_pct'],
            stats['profit_factor'],
            stats['roi_pct'],
            stats['max_drawdown_pct'],
            stats['start_balance'],
            stats['end_balance'],
        )

    def trade(self, pair, direction, size, latest_row, entry, stop):
        self.fetch_account_balance()
        if not self.demo_mode:
            logging.info("LIVE trade function NOT IMPLEMENTED in this demo bot.")
            return

        self.daily_trade_count += 1
        pnl_multiplier = np.random.normal(loc=0.05, scale=0.35)
        profit = size * pnl_multiplier
        self.balance += profit
        self.last_trade_time = datetime.datetime.now(datetime.timezone.utc)

        self.log_trade_memory(latest_row, direction, profit)
        self.log_trade_journal(
            {
                'timestamp': self.last_trade_time.isoformat(),
                'pair': pair,
                'direction': direction,
                'entry': round(float(entry), 8),
                'stop': round(float(stop), 8),
                'size': round(float(size), 6),
                'profit': round(float(profit), 6),
                'balance_after': round(float(self.balance), 6),
                'source': 'live_demo',
                'won': 1 if profit > 0 else 0,
            }
        )

        if self.ai_retrain_every_n_trades > 0 and self.daily_trade_count % self.ai_retrain_every_n_trades == 0:
            self.retrain_ai_from_memory()

        outcome = "Profit" if profit > 0 else "Loss"
        logging.info(
            "TRADE: %s | %s | Size %.4f | %s $%.2f | New Balance: $%.2f",
            pair,
            direction.upper(),
            size,
            outcome,
            abs(profit),
            self.balance,
        )

    def run_backtest(self, bars=800, payout=0.82):
        logging.info("=== Running historical backtest (%s bars, payout %.2f) ===", bars, payout)
        for pair in self.pairs:
            try:
                result = self.backtest_pair(pair, bars=bars, payout=payout)
            except RuntimeError as exc:
                logging.warning("Skipping %s backtest: %s", pair, exc)
                continue

            logging.info(
                "%s | trades=%s wins=%s losses=%s win_rate=%s%% roi=%s%% ending=$%s max_dd=%s%% pf=%s",
                result['symbol'],
                result['trades'],
                result['wins'],
                result['losses'],
                result['win_rate_pct'],
                result['roi_pct'],
                result['ending_balance'],
                result['max_drawdown_pct'],
                result['profit_factor'],
            )

    def backtest_pair(self, symbol, bars=800, payout=0.82):
        df = self.get_market_data(symbol, limit=bars)
        equity = float(self.start_of_day_balance)
        rows = []

        for i in range(60, len(df) - 1):
            window = df.iloc[: i + 1]
            candle_hour = int(window['timestamp'].iloc[-1].hour)
            direction = self.analyze_ict(window, session_hour=candle_hour, log_signal=False)
            if not direction:
                continue

            entry = float(df['close'].iloc[i])
            nxt = float(df['close'].iloc[i + 1])
            risk_amount = equity * (self.risk_pct / 100)

            won = (direction == 'buy' and nxt > entry) or (direction == 'sell' and nxt < entry)
            profit = risk_amount * payout if won else -risk_amount
            equity += profit

            rows.append(
                {
                    'timestamp': df['timestamp'].iloc[i + 1].isoformat(),
                    'pair': symbol,
                    'direction': direction,
                    'entry': entry,
                    'stop': np.nan,
                    'size': risk_amount,
                    'profit': profit,
                    'balance_after': equity,
                    'source': 'backtest',
                    'won': 1 if won else 0,
                }
            )

        if not rows:
            return {
                'symbol': symbol,
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate_pct': 0.0,
                'roi_pct': 0.0,
                'ending_balance': round(equity, 2),
                'max_drawdown_pct': 0.0,
                'profit_factor': 0.0,
            }

        trades_df = pd.DataFrame(rows)
        for _, rec in trades_df.iterrows():
            self.log_trade_journal(rec.to_dict())

        stats = self.performance_metrics(trades_df)
        return {
            'symbol': symbol,
            'trades': stats['trades'],
            'wins': stats['wins'],
            'losses': stats['losses'],
            'win_rate_pct': stats['win_rate_pct'],
            'roi_pct': stats['roi_pct'],
            'ending_balance': stats['end_balance'],
            'max_drawdown_pct': stats['max_drawdown_pct'],
            'profit_factor': stats['profit_factor'],
        }

    def run(self):
        logging.info("=== Pocket Option ICT 5-Star Bot (Upgraded Edition) ===")
        logging.info("=== Pocket Option ICT + AI Hybrid Bot ===")
        while True:
            self.reset_daily_trades()

            if self.daily_trade_count >= self.max_trades:
                logging.info(f"Trade cap hit for today ({self.max_trades} trades). Sleeping until tomorrow.")
                time.sleep(60 * 60 * 2)
                logging.info("Trade cap hit for today (%s trades). Sleeping.", self.max_trades)
                time.sleep(60 * 30)
                continue

            if self.is_daily_drawdown_breached():
                time.sleep(60 * 30)
                continue

            if not self.is_cooldown_over():
                logging.info("Trade cooldown active. Waiting for next cycle.")
                time.sleep(60)
                continue

            self.ensure_api_freshness()
            for pair in self.pairs:
                df = self.get_market_data(pair)
                direction = self.analyze_ict(df)
                try:
                    df = self.get_market_data(pair, self.market_data_limit)
                except RuntimeError as exc:
                    logging.warning("Skipping %s: %s", pair, exc)
                    continue

                candle_hour = int(df['timestamp'].iloc[-1].hour)
                direction = self.analyze_ict(df, session_hour=candle_hour)
                if direction:
                    entry = df['close'].iloc[-1]
                    # Stop is 1 ATR below/above for risk calculation (simplified)
                    atr = df['high'].rolling(14).max().iloc[-1] - df['low'].rolling(14).min().iloc[-1]
                    entry = float(df['close'].iloc[-1])
                    atr = float(df['atr'].iloc[-1])
                    if np.isnan(atr) or atr <= 0:
                        logging.warning("Skipping %s: ATR unavailable.", pair)
                        continue

                    stop = entry - atr if direction == 'buy' else entry + atr
                    size = self.risk_size(entry, stop)
                    if self.daily_trade_count < self.max_trades:
                        self.trade(pair, direction, size)
                if self.daily_trade_count >= self.max_trades:
                    latest_row = df.iloc[-1]
                    self.trade(pair, direction, size, latest_row, entry, stop)

                if self.daily_trade_count >= self.max_trades or self.is_daily_drawdown_breached():
                    break
            time.sleep(120)  # align with 2min candle (demo frequency)

            time.sleep(120)


def parse_args():
    parser = argparse.ArgumentParser(description="Pocket Option ICT + AI bot")
    parser.add_argument('--mode', choices=['run', 'backtest', 'report'], default='run')
    parser.add_argument('--bars', type=int, default=800, help='Backtest candles per pair.')
    parser.add_argument('--payout', type=float, default=0.82, help='Estimated payout multiplier for wins.')
    parser.add_argument('--source', choices=['all', 'live_demo', 'backtest'], default='all', help='Report source filter.')
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