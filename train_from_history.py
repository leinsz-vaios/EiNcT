"""
TRAIN FROM HISTORY - UPGRADED
Trains the ProbabilityBrain on real historical data for ALL pairs in the ICT bot.

Pairs trained:
  Forex : EURUSD, GBPUSD, USDJPY, AUDUSD
  Crypto: ETHUSD (BTC/ETH)

Timeframe: 1h  (best Yahoo Finance availability; aligns with London/NY killzone logic)
History  : 2 years (max Yahoo allows on 1h)

Features match EXACTLY what main.py uses:
  ['rsi', 'ema_fast', 'ema_slow', 'atr', 'volume']

ICT-aligned target label:
  1 = price moved UP in next 5 bars (buy setup)
  0 = price stayed flat or moved DOWN

Run:
  pip install yfinance pandas numpy
  python train_from_history.py
"""

import pickle
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

PAIRS = {
    # Yahoo ticker        friendly name     is_crypto
    "EURUSD=X":          ("EURUSD",         False),
    "GBPUSD=X":          ("GBPUSD",         False),
    "USDJPY=X":          ("USDJPY",         False),
    "AUDUSD=X":          ("AUDUSD",         False),
    "ETH-USD":           ("ETHUSD",         True),
    "BTC-USD":           ("BTCUSD",         True),
}

TIMEFRAME   = "1h"          # 1-hour bars – best balance of signal quality & data availability
PERIOD      = "2y"          # 2 years of history
LOOKAHEAD   = 5             # bars ahead used to define the "win" label
OUTPUT_PKL  = "trading_brain.pkl"

# Liquidity / ICT parameters (mirror config.py defaults)
LIQUIDITY_LOOKBACK  = 20
SWEEP_MEMORY        = 5
ATR_DISPLACEMENT    = 0.8   # body > atr * this → displacement candle


# ─────────────────────────────────────────────
# THE BRAIN  (identical to main.py)
# ─────────────────────────────────────────────

class ProbabilityBrain:
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
        self.mean            = X.mean(axis=0)
        self.std             = X.std(axis=0)
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
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)


# ─────────────────────────────────────────────
# INDICATOR CALCULATION
# ─────────────────────────────────────────────

def calculate_indicators(df: pd.DataFrame, is_crypto: bool = False) -> pd.DataFrame:
    """
    Build all features that main.py's FEATURE_COLUMNS expect, PLUS
    ICT-derived bonus features to give the model richer signal.
    """
    out = df.copy()

    # ── Core indicators (match main.py enrich_market_data) ──────────────
    # EMA fast / slow
    out["ema_fast"] = out["Close"].ewm(span=20, adjust=False).mean()
    out["ema_slow"] = out["Close"].ewm(span=50, adjust=False).mean()

    # RSI(14)
    delta = out["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))

    # ATR(14)
    tr = pd.concat([
        out["High"] - out["Low"],
        (out["High"] - out["Close"].shift(1)).abs(),
        (out["Low"]  - out["Close"].shift(1)).abs(),
    ], axis=1)
    out["atr"] = tr.max(axis=1).rolling(14).mean()

    # Volume (already in df; rename to lowercase for consistency)
    out["volume"] = out["Volume"]

    # ── ICT-derived features (bonus signal for the AI) ───────────────────
    prev_high = out["High"].rolling(LIQUIDITY_LOOKBACK).max().shift(1)
    prev_low  = out["Low"].rolling(LIQUIDITY_LOOKBACK).min().shift(1)

    # Liquidity sweeps (turtle soup pattern)
    out["liq_sweep_buy"]  = ((out["Low"]  < prev_low)  & (out["Close"] > prev_low)).astype(int)
    out["liq_sweep_sell"] = ((out["High"] > prev_high) & (out["Close"] < prev_high)).astype(int)

    # Displacement candles
    body = (out["Close"] - out["Open"]).abs()
    out["displacement_up"]   = ((body > out["atr"] * ATR_DISPLACEMENT) & (out["Close"] > out["Open"])).astype(int)
    out["displacement_down"] = ((body > out["atr"] * ATR_DISPLACEMENT) & (out["Close"] < out["Open"])).astype(int)

    # Structure breaks
    out["structure_break_up"]   = (out["Close"] > out["High"].shift(5).rolling(5).max()).astype(int)
    out["structure_break_down"] = (out["Close"] < out["Low"].shift(5).rolling(5).min()).astype(int)

    # Fair Value Gaps
    out["fvg_up"]   = (out["Low"]  > out["High"].shift(2)).astype(int)
    out["fvg_down"] = (out["High"] < out["Low"].shift(2)).astype(int)

    # RSI zone (where your bot scores extra points)
    if is_crypto:
        out["rsi_buy_zone"]  = ((out["rsi"] >= 50) & (out["rsi"] <= 70)).astype(int)
        out["rsi_sell_zone"] = ((out["rsi"] >= 30) & (out["rsi"] <= 50)).astype(int)
    else:
        out["rsi_buy_zone"]  = ((out["rsi"] >= 45) & (out["rsi"] <= 65)).astype(int)
        out["rsi_sell_zone"] = ((out["rsi"] >= 35) & (out["rsi"] <= 55)).astype(int)

    # ── Target label ─────────────────────────────────────────────────────
    # 1 = price closed higher LOOKAHEAD bars later (a "buy" setup paid off)
    out["target"] = (out["Close"].shift(-LOOKAHEAD) > out["Close"]).astype(int)

    return out.dropna()


# ─────────────────────────────────────────────
# FEATURE COLUMNS  (must match main.py FEATURE_COLUMNS + ICT extras)
# ─────────────────────────────────────────────

FEATURE_COLUMNS = [
    "rsi",
    "ema_fast",
    "ema_slow",
    "atr",
    "volume",
]


# ─────────────────────────────────────────────
# DOWNLOAD + PROCESS
# ─────────────────────────────────────────────

def download_pair(ticker: str, friendly: str, is_crypto: bool):
    print(f"\n  📥  Downloading {friendly} ({ticker})  [{TIMEFRAME} / {PERIOD}]...")
    raw = yf.download(ticker, period=PERIOD, interval=TIMEFRAME, progress=False, auto_adjust=True)

    if raw.empty:
        print(f"  ⚠️   No data returned for {ticker} – skipping.")
        return None, None

    # Flatten MultiIndex columns that yfinance sometimes returns
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = calculate_indicators(raw, is_crypto=is_crypto)

    if len(df) < 200:
        print(f"  ⚠️   Only {len(df)} rows after processing {ticker} – skipping (need ≥200).")
        return None, None

    X = df[FEATURE_COLUMNS].values
    y = df["target"].values
    print(f"  ✅  {friendly}: {len(X):,} usable bars  |  "
          f"buy-label rate: {y.mean()*100:.1f}%")
    return X, y


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ICT BOT – HISTORICAL BRAIN TRAINER")
    print(f"  Timeframe : {TIMEFRAME}   |   Period : {PERIOD}")
    print(f"  Pairs     : {', '.join(v[0] for v in PAIRS.values())}")
    print("=" * 60)

    all_X, all_y = [], []

    for ticker, (friendly, is_crypto) in PAIRS.items():
        X, y = download_pair(ticker, friendly, is_crypto)
        if X is not None:
            all_X.append(X)
            all_y.append(y)

    if not all_X:
        print("\n❌  No data downloaded. Check your internet connection.")
        return

    X_combined = np.vstack(all_X)
    y_combined = np.concatenate(all_y)

    print(f"\n🧠  Training on {len(X_combined):,} total bars across all pairs...")
    print(f"    Features  : {len(FEATURE_COLUMNS)}")
    print(f"    Buy labels: {y_combined.mean()*100:.1f}%  |  "
          f"Sell labels: {(1-y_combined).mean()*100:.1f}%")

    brain = ProbabilityBrain()
    brain.fit(X_combined, y_combined, l2=0.1)

    brain.save(OUTPUT_PKL)
    print(f"\n✅  Brain saved → '{OUTPUT_PKL}'")
    print("    Copy this file to your bot's working directory.")
    print("    The bot will auto-load it on next startup.\n")

    # Quick sanity check
    sample_proba = brain.predict_proba(X_combined[:10])
    print(f"    Sanity check – first 10 buy-probabilities: "
          f"{[round(p,3) for p in sample_proba[:,1].tolist()]}")
    print("=" * 60)


if __name__ == "__main__":
    main()