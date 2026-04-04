import argparse
import logging

import ccxt
import pandas as pd
from ccxt.base.errors import NetworkError as CCXTNetworkError

import config
from main import PocketOptionBot, ProbabilityBrain

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')


def fetch_training_data(markets, bars):
    exchange = ccxt.kraken()
    frames = []

    for market in markets:
        logging.info("Downloading %s candles for %s", bars, market)
        try:
            candles = exchange.fetch_ohlcv(market, timeframe='1m', limit=bars)
        except CCXTNetworkError as exc:
            logging.warning("Skipping %s due to network error: %s", market, exc)
            continue

        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        frames.append(df)

    if not frames:
        raise RuntimeError("No training data downloaded. Check network and retry.")

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values('timestamp').reset_index(drop=True)
    return out


def train_and_validate(df, horizon=1, split_ratio=0.8):
    bot = PocketOptionBot()
    enriched = bot.enrich_market_data(df)
    enriched['target'] = (enriched['close'].shift(-horizon) > enriched['close']).astype(int)

    usable = enriched.dropna(subset=bot.FEATURE_COLUMNS + ['target']).copy()
    split_idx = int(len(usable) * split_ratio)
    split_idx = max(50, min(split_idx, len(usable) - 1))

    train_df = usable.iloc[:split_idx]
    test_df = usable.iloc[split_idx:]

    model = ProbabilityBrain()
    model.fit(train_df[bot.FEATURE_COLUMNS].values, train_df['target'].values)

    train_pred = (model.predict_proba(train_df[bot.FEATURE_COLUMNS].values)[:, 1] >= 0.5).astype(int)
    test_pred = (model.predict_proba(test_df[bot.FEATURE_COLUMNS].values)[:, 1] >= 0.5).astype(int)

    train_acc = float((train_pred == train_df['target'].values).mean())
    test_acc = float((test_pred == test_df['target'].values).mean())

    return model, len(train_df), len(test_df), train_acc, test_acc


def parse_args():
    parser = argparse.ArgumentParser(description='Train AI model for PocketOptionBot')
    parser.add_argument('--bars', type=int, default=2000)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--split-ratio', type=float, default=0.8)
    parser.add_argument('--model-path', default=config.AI_MODEL_PATH)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    markets = [PocketOptionBot.symbol_to_market(p) for p in config.PAIRS]

    raw = fetch_training_data(markets, bars=args.bars)
    model, train_rows, test_rows, train_acc, test_acc = train_and_validate(
        raw,
        horizon=args.horizon,
        split_ratio=args.split_ratio,
    )

    model.save(args.model_path)
    logging.info("Saved model to %s", args.model_path)
    logging.info("Train rows=%s | Test rows=%s", train_rows, test_rows)
    logging.info("Train acc=%.3f | Out-of-sample acc=%.3f", train_acc, test_acc)
