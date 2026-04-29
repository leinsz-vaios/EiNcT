import yfinance as yf
import pandas as pd
import numpy as np
import pickle

# --- The Brain Class (Must stay identical to your bot's code) ---
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

    def save(self, path):
        with open(path, 'wb') as fh:
            pickle.dump(self, fh)

# --- Indicator Calculations ---
def calculate_indicators(df):
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # EMAs
    df['ema_fast'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['ema_slow'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # ATR (True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['atr'] = ranges.max(axis=1).rolling(14).mean()
    
    # Target: Did price go UP in the next 5 bars? (Label 1 if yes, 0 if no)
    # This teaches the AI to recognize "Buy" setups
    df['target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
    
    return df.dropna()

# --- Main Execution ---
def main():
    symbol = "GBPUSD=X" # Yahoo Finance format for GBP/USD
    print(f"📥 Downloading real historical data for {symbol}...")
    
    # Get 2 years of 1-hour data
    data = yf.download(symbol, period="2y", interval="1h")
    
    print("⚙️ Processing indicators...")
    df = calculate_indicators(data)
    
    # Prepare Features (Match the bot's expected input)
    features = ['rsi', 'ema_fast', 'ema_slow', 'atr', 'Volume']
    X = df[features].values
    y = df['target'].values
    
    print(f"🧠 Training on {len(X)} real market hours...")
    brain = ProbabilityBrain()
    brain.fit(X, y)
    
    brain.save('trading_brain.pkl')
    print("✅ Real-world trained model saved as 'trading_brain.pkl'")

if __name__ == "__main__":
    main()