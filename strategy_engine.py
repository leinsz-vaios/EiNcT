"""
strategy_engine.py
──────────────────
Drop-in strategy dispatcher for PocketOptionBot.
Reads strategy_knowledge.json and runs each enabled strategy's
signal logic. Returns the highest-confidence direction & which
strategy fired it.

Usage (in main.py):
    from strategy_engine import StrategyEngine
    # In __init__:
    self.strategy_engine = StrategyEngine(config.STRATEGY_KNOWLEDGE_PATH)
    # Replace analyze_ict() calls with:
    result = self.strategy_engine.run(df, hour_utc=..., htf_bias=...)
    direction = result['direction']    # 'buy' | 'sell' | None
    signal_tag = result['strategy']    # 'LPR' | 'TS' | 'SMR' | 'SB' | None
"""

import json
import logging
import os
import datetime

import numpy as np
import pandas as pd


# ─── helpers ──────────────────────────────────────────────────────────────────

def _swing_highs(series: pd.Series, n: int = 5) -> pd.Series:
    """Returns boolean mask where swing highs exist."""
    left  = series.rolling(n).max()
    right = series[::-1].rolling(n).max()[::-1]
    return (series == left) & (series == right)


def _swing_lows(series: pd.Series, n: int = 5) -> pd.Series:
    left  = series.rolling(n).min()
    right = series[::-1].rolling(n).min()[::-1]
    return (series == left) & (series == right)


def _detect_fvg(df: pd.DataFrame, direction: str, min_pips: float = 3.0) -> bool:
    """
    Three-candle Fair Value Gap detection.
    Bullish FVG: candle[i-2].high < candle[i].low  (gap upward)
    Bearish FVG: candle[i-2].low  > candle[i].high (gap downward)
    Checks last `max_age` candles.
    """
    max_age = 10
    window = df.tail(max_age)
    if len(window) < 3:
        return False
    avg_price = float(df["close"].iloc[-1])
    pip = avg_price * 0.0001  # approximate pip size

    for i in range(2, len(window)):
        c0 = window.iloc[i - 2]
        c2 = window.iloc[i]
        if direction == "buy":
            gap = c2["low"] - c0["high"]
        else:
            gap = c0["low"] - c2["high"]
        if gap >= min_pips * pip:
            return True
    return False


def _detect_sweep(df: pd.DataFrame, level: float, direction: str,
                  min_pips: float = 3.0, max_candles: int = 3) -> bool:
    """
    Checks if price spiked through `level` and closed back within last
    `max_candles` bars (i.e. a wick-based sweep).
    direction='above' → spike above level then closed below (sweep of resistance)
    direction='below' → spike below level then closed above (sweep of support)
    """
    window = df.tail(max_candles + 2)
    if len(window) < 3:
        return False
    avg_price = float(df["close"].iloc[-1])
    pip = avg_price * 0.0001

    for i in range(1, len(window)):
        bar = window.iloc[i]
        if direction == "above":
            breached = bar["high"] > level + min_pips * pip
            closed_back = bar["close"] < level
        else:
            breached = bar["low"] < level - min_pips * pip
            closed_back = bar["close"] > level
        if breached and closed_back:
            return True
    return False


def _mss_confirmed(df: pd.DataFrame, direction: str, swing_bars: int = 5) -> bool:
    """
    Market Structure Shift: latest close breaks the most recent
    opposite swing point.
    direction='buy'  → latest close > recent swing high
    direction='sell' → latest close < recent swing low
    """
    if len(df) < swing_bars * 2:
        return False
    latest_close = float(df["close"].iloc[-1])
    if direction == "buy":
        recent_swing = float(df["high"].tail(swing_bars * 2).max())
        return latest_close > recent_swing
    else:
        recent_swing = float(df["low"].tail(swing_bars * 2).min())
        return latest_close < recent_swing


def _rsi_extreme(rsi_val: float, direction: str) -> bool:
    """RSI at extreme zone supporting the trade direction."""
    if direction == "buy"  and rsi_val <= 40:
        return True
    if direction == "sell" and rsi_val >= 60:
        return True
    return False


def _ema_aligned(df: pd.DataFrame, direction: str,
                 fast_period: int = 20, slow_period: int = 50) -> bool:
    fast = float(df["close"].ewm(span=fast_period, adjust=False).mean().iloc[-1])
    slow = float(df["close"].ewm(span=slow_period, adjust=False).mean().iloc[-1])
    if direction == "buy":
        return fast > slow
    return fast < slow


def _in_ny_window(hour_ny: int, windows: list) -> bool:
    """Check if current NY hour falls in any enabled time window."""
    for w in windows:
        if w.get("enabled", True) and w["start_hour"] <= hour_ny < w["end_hour"]:
            return True
    return False


def _utc_to_ny_hour(hour_utc: int) -> int:
    """Approximate UTC → New York offset (EST = UTC-5, EDT = UTC-4).
    Uses a fixed -5 offset; adjust for DST in production."""
    return (hour_utc - 5) % 24


def _in_killzone(hour_utc: int, killzones: dict) -> bool:
    for _name, kz in killzones.items():
        if kz["start_hour_utc"] <= hour_utc < kz["end_hour_utc"]:
            return True
    return False


# ─── main engine ──────────────────────────────────────────────────────────────

class StrategyEngine:
    """
    Loads strategy_knowledge.json and exposes .run() which evaluates all
    enabled strategies against current market data, returning the best signal.
    """

    def __init__(self, knowledge_path: str):
        self.knowledge_path = knowledge_path
        self.knowledge = {}
        self.strategies = {}
        self._load()

    # ── loading ──────────────────────────────────────────────────────────────

    def _load(self):
        if not os.path.exists(self.knowledge_path):
            logging.warning("[StrategyEngine] knowledge file not found: %s", self.knowledge_path)
            return
        with open(self.knowledge_path, "r", encoding="utf-8") as fh:
            self.knowledge = json.load(fh)
        for item in self.knowledge.get("strategy_items", []):
            if item.get("enabled", True):
                self.strategies[item["id"]] = item
        logging.info("[StrategyEngine] Loaded %d strategies: %s",
                     len(self.strategies), list(self.strategies.keys()))

    def reload(self):
        """Re-read the JSON from disk (call if you hot-update the file)."""
        self._load()

    # ── public API ───────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame, hour_utc: int = None,
            htf_bias: str = None, min_signal_score: int = None) -> dict:
        """
        Evaluate all enabled strategies.

        Parameters
        ----------
        df            : enriched OHLCV DataFrame (must have ema_fast, ema_slow, rsi, atr)
        hour_utc      : current UTC hour (int); defaults to now
        htf_bias      : 'buy' | 'sell' | None  (from timeframe_bias())
        min_signal_score : override the per-strategy min; None = use each strategy's own

        Returns
        -------
        dict with keys:
            direction  : 'buy' | 'sell' | None
            strategy   : strategy signal_tag string or None
            score      : int score of the winning signal
            details    : dict of per-strategy scores
        """
        if hour_utc is None:
            hour_utc = datetime.datetime.now(datetime.timezone.utc).hour

        if len(df) < 30:
            return {"direction": None, "strategy": None, "score": 0, "details": {}}

        best = {"direction": None, "strategy": None, "score": 0, "details": {}}

        for sid, strategy in self.strategies.items():
            p = strategy.get("parameters", {})
            scoring = p.get("scoring", {})
            threshold = min_signal_score if min_signal_score is not None \
                        else scoring.get("min_score_to_trade", 4)

            try:
                if sid == "liquidity_purge_revert":
                    direction, score = self._score_lpr(df, p, scoring, hour_utc, htf_bias)
                elif sid == "turtle_soup":
                    direction, score = self._score_ts(df, p, scoring, htf_bias)
                elif sid == "smart_money_reversal":
                    direction, score = self._score_smr(df, p, scoring, htf_bias)
                elif sid == "silver_bullet":
                    direction, score = self._score_sb(df, p, scoring, hour_utc, htf_bias)
                else:
                    continue
            except Exception as exc:
                logging.warning("[StrategyEngine] Error in %s: %s", sid, exc)
                continue

            tag = strategy.get("signal_tag", sid)
            best["details"][tag] = {"direction": direction, "score": score}

            if direction and score >= threshold and score > best["score"]:
                best["direction"] = direction
                best["strategy"]  = tag
                best["score"]     = score

        if best["direction"]:
            logging.info("[StrategyEngine] Signal=%s strategy=%s score=%d details=%s",
                         best["direction"], best["strategy"], best["score"], best["details"])
        return best

    # ── strategy scorers ─────────────────────────────────────────────────────

    def _score_lpr(self, df, p, scoring, hour_utc, htf_bias):
        """Liquidity Purge & Revert scoring."""
        sweep_p    = p.get("sweep", {})
        confirm_p  = p.get("confirmation", {})
        kz         = confirm_p.get("killzones", {})

        latest     = df.iloc[-1]
        rsi        = float(latest.get("rsi", 50))
        swing_bars = confirm_p.get("mss_swing_bars", 5)
        min_fvg    = confirm_p.get("fvg_min_size_pips", 3)

        # Pick direction from recent swing structure
        recent_high = float(df["high"].tail(swing_bars * 2).max())
        recent_low  = float(df["low"].tail(swing_bars * 2).min())
        close       = float(latest["close"])

        # Determine candidate direction
        if close < recent_low * 1.002:    # near recent low → potential long after sweep
            direction = "buy"
        elif close > recent_high * 0.998: # near recent high → potential short after sweep
            direction = "sell"
        else:
            direction = "buy" if htf_bias == "buy" else "sell" if htf_bias == "sell" else None

        if direction is None:
            return None, 0

        score = 0
        # Check sweep
        level = recent_low if direction == "buy" else recent_high
        side  = "below" if direction == "buy" else "above"
        if _detect_sweep(df, level, side,
                         min_pips=sweep_p.get("min_breach_pips", 3),
                         max_candles=sweep_p.get("max_candles_to_confirm", 3)):
            score += scoring.get("sweep_detected_points", 3)

        # MSS confirmation
        if _mss_confirmed(df, direction, swing_bars):
            score += scoring.get("mss_confirmed_points", 3)

        # Killzone bonus
        if confirm_p.get("require_killzone", False) or _in_killzone(hour_utc, kz):
            score += scoring.get("killzone_active_points", 1)

        # FVG present
        if _detect_fvg(df, direction, min_pips=min_fvg):
            score += scoring.get("fvg_present_points", 2)

        # HTF bias aligned
        if htf_bias == direction:
            score += scoring.get("htf_bias_aligned_points", 2)

        return (direction, score) if score > 0 else (None, 0)

    def _score_ts(self, df, p, scoring, htf_bias):
        """Turtle Soup scoring."""
        ch_period  = p.get("channel", {}).get("period", 20)
        min_bars   = p.get("channel", {}).get("min_bars_since_last_break", 3)

        if len(df) < ch_period + min_bars:
            return None, 0

        ch_high = float(df["high"].tail(ch_period).max())
        ch_low  = float(df["low"].tail(ch_period).min())
        close   = float(df["close"].iloc[-1])
        rsi     = float(df["rsi"].iloc[-1]) if "rsi" in df.columns else 50.0

        # Count bars since last channel break
        breaks_below = (df["low"].tail(ch_period) < ch_low).sum()
        breaks_above = (df["high"].tail(ch_period) > ch_high).sum()

        direction = None
        score = 0

        if close < ch_low and breaks_below <= 2:
            direction = "buy"   # false break below → expect bounce back up
            score += scoring.get("channel_break_detected_points", 3)
            # Plus-One: bar closed outside channel
            if p.get("variant") == "plus_one" and close < ch_low:
                score += scoring.get("close_outside_channel_points", 2)
            # RSI oversold
            if rsi < 35:
                score += scoring.get("rsi_extreme_points", 1)

        elif close > ch_high and breaks_above <= 2:
            direction = "sell"  # false break above → expect bounce back down
            score += scoring.get("channel_break_detected_points", 3)
            if p.get("variant") == "plus_one" and close > ch_high:
                score += scoring.get("close_outside_channel_points", 2)
            if rsi > 65:
                score += scoring.get("rsi_extreme_points", 1)

        if direction is None:
            return None, 0

        # Bars-elapsed bonus
        score += scoring.get("min_bars_elapsed_points", 2)

        # Volume spike
        if "vol_ma" in df.columns:
            vol_now = float(df["volume"].iloc[-1])
            vol_avg = float(df["vol_ma"].iloc[-1])
            if vol_now > vol_avg * 1.5:
                score += scoring.get("volume_spike_points", 1)

        return (direction, score) if score > 0 else (None, 0)

    def _score_smr(self, df, p, scoring, htf_bias):
        """Smart Money Reversal scoring."""
        sr_p   = p.get("stop_run", {})
        fvg_p  = p.get("fvg", {})
        mss_p  = p.get("mss", {})
        htf_p  = p.get("htf_analysis", {})

        if len(df) < 20:
            return None, 0

        rsi    = float(df["rsi"].iloc[-1]) if "rsi" in df.columns else 50.0
        close  = float(df["close"].iloc[-1])

        # Determine direction from HTF bias or EMA cross
        fast_p = htf_p.get("ema_period_fast", 20)
        slow_p = htf_p.get("ema_period_slow", 50)
        fast   = float(df["close"].ewm(span=fast_p, adjust=False).mean().iloc[-1])
        slow   = float(df["close"].ewm(span=slow_p, adjust=False).mean().iloc[-1])

        if htf_bias == "buy" or fast > slow:
            direction = "buy"
        elif htf_bias == "sell" or fast < slow:
            direction = "sell"
        else:
            return None, 0

        score = 0
        lookback = p.get("liquidity_pools", {}).get("lookback_bars", 50)
        pool_level = (float(df["low"].tail(lookback).min()) if direction == "buy"
                      else float(df["high"].tail(lookback).max()))

        sweep_side = "below" if direction == "buy" else "above"
        if _detect_sweep(df, pool_level, sweep_side,
                         min_pips=sr_p.get("min_breach_pips", 5),
                         max_candles=sr_p.get("max_candles_to_recover", 4)):
            score += scoring.get("stop_run_detected_points", 3)

        if _detect_fvg(df, direction, min_pips=fvg_p.get("min_size_pips", 3)):
            score += scoring.get("fvg_formed_points", 3)

        if _mss_confirmed(df, direction, mss_p.get("swing_bars", 5)):
            score += scoring.get("mss_confirmed_points", 3)

        if _ema_aligned(df, direction, fast_p, slow_p):
            score += scoring.get("htf_ema_aligned_points", 2)

        if _rsi_extreme(rsi, direction):
            score += scoring.get("rsi_extreme_at_sweep_points", 1)

        return (direction, score) if score > 0 else (None, 0)

    def _score_sb(self, df, p, scoring, hour_utc, htf_bias):
        """Silver Bullet scoring."""
        windows   = p.get("time_windows_ny", [])
        sweep_p   = p.get("sweep", {})
        fvg_p     = p.get("fvg", {})
        bias_p    = p.get("daily_bias_filter", {})

        hour_ny = _utc_to_ny_hour(hour_utc)

        # Must be inside a Silver Bullet window
        if not _in_ny_window(hour_ny, windows):
            return None, 0

        score = scoring.get("in_time_window_points", 3)

        if len(df) < 20:
            return None, 0

        rsi   = float(df["rsi"].iloc[-1]) if "rsi" in df.columns else 50.0
        close = float(df["close"].iloc[-1])

        # Identify nearest SSL (buy setup) or BSL (sell setup)
        lookback = p.get("liquidity_zones", {}).get("lookback_bars", 20)
        ssl = float(df["low"].tail(lookback).min())   # sell-side liquidity
        bsl = float(df["high"].tail(lookback).max())  # buy-side liquidity

        dist_to_ssl = abs(close - ssl)
        dist_to_bsl = abs(close - bsl)

        # Determine setup direction: closer to SSL → potential long; closer to BSL → short
        if dist_to_ssl < dist_to_bsl:
            direction = "buy"
            level, side = ssl, "below"
        else:
            direction = "sell"
            level, side = bsl, "above"

        # Filter by daily bias
        if bias_p.get("enabled", True):
            ema_p = bias_p.get("ema_period", 50)
            ema50 = float(df["close"].ewm(span=ema_p, adjust=False).mean().iloc[-1])
            if direction == "buy"  and close < ema50:
                return None, 0
            if direction == "sell" and close > ema50:
                return None, 0
            if htf_bias and htf_bias != direction:
                return None, 0

        # Sweep detected
        if _detect_sweep(df, level, side,
                         min_pips=sweep_p.get("min_breach_pips", 2),
                         max_candles=sweep_p.get("max_candles_to_recover", 3)):
            score += scoring.get("sweep_detected_points", 3)
        else:
            return None, 0  # sweep is mandatory for Silver Bullet

        # FVG formed after sweep
        if _detect_fvg(df, direction, min_pips=fvg_p.get("min_size_pips", 2)):
            score += scoring.get("fvg_formed_points", 3)
        else:
            return None, 0  # FVG is mandatory for Silver Bullet

        # HTF bias aligned
        if htf_bias == direction:
            score += scoring.get("daily_bias_aligned_points", 2)

        # Optional MSS bonus
        if _mss_confirmed(df, direction):
            score += scoring.get("mss_confirmed_points", 1)

        return (direction, score) if score > 0 else (None, 0)