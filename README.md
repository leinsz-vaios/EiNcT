diff --git a/README.md b/README.md
index b4a6d0c3250fd5aec8245ca4022ded25e6af5de4..3f93c5b64c905297cbb51d997f3641415b7c6ba1 100644
--- a/README.md
+++ b/README.md
@@ -1,83 +1,104 @@
 # Pocket Option ICT "5 Star" Pro Bot
 
-_A fully automated, high-class ICT trading bot for Pocket Option, entirely configurable and built for precision trading._  
-**Author**: leinsz-vaios
+_A data-driven ICT trading bot with an optional AI filter that checks both historical patterns and present market context before a trade._
 
 ---
 
-## ⚡ Features
-- ✓ Fair Value Gap (FVG), Order Block, and Market Structure logic
-- ✓ Automatic Pocket Option API endpoint freshness check
-- ✓ Per-account API key/password – no hardcoded secrets!
-- ✓ Up to **3 trades/day**, all risk-managed
-- ✓ Full session filtering (London/NY only)
-- ✓ Runs in demo or real mode (set in `.env`)
-- ✓ Simple, clear logs for every action
+## ⚡ What makes this company-grade
+- ICT + indicator scoring (market structure, FVG, EMA, RSI, ATR, volume).
+- AI probability veto before execution (`AI_MIN_BUY_PROB`, `AI_MAX_SELL_PROB`).
+- Trade memory + periodic retraining from outcomes.
+- Risk guardrails: max trades/day, drawdown cap, cooldown.
+- Persistent trade journal for auditable reporting.
+- Backtest mode + performance report mode.
 
 ---
 
-## 🛠️ Setup Instructions
+## 🛠️ Setup
 
-**1. Get the Code**
-```bash
-git clone https://github.com/leinsz-vaios/ict-trading-bot.git
-cd ict-trading-bot
-```
-
-**2. Configure Python Environment**
 ```bash
 python -m venv venv
-source venv/bin/activate            # On macOS/Linux
-# OR: venv\\Scripts\\activate       # On Windows
-```
-
-**3. Install Requirements**
-```bash
+source venv/bin/activate
 pip install -r requirements.txt
 ```
 
-**4. Configure Your `.env` File**
-Copy `.env.example` to `.env` and fill in your Pocket Option email/password.
+---
+
+## 1) Train AI on past data
+
 ```bash
-cp .env.example .env
+python train_ai.py --bars 2000 --horizon 1 --split-ratio 0.8
 ```
-Edit `.env` to set your credentials, risk, and config.
+
+What you get:
+- Train accuracy
+- Out-of-sample (test) accuracy
+- Saved model file (`trading_brain.pkl`)
 
 ---
 
-## 🚀 Run the Bot
+## 2) Run bot on present market data
 
 ```bash
-python main.py
+python main.py --mode run
 ```
 
-_The bot will run, trading a maximum of **3 trades per day** with full ICT analysis. See logs for trade actions and current balance._
-
 ---
 
-## ⚠️ Important Notes
+## 3) Know your REAL win rate (important)
 
-- **Profit Guarantees:** No trading system can guarantee an 85% win rate; real markets have risk. Test in demo mode first.
-- **Codespaces & 24/7:** Codespaces pause after inactivity! For true 24/7, consider deploying your bot on a VPS or a cloud service.
-- **API Automation:** The bot auto-fetches the API endpoint each session. Update your `.env` only for new credentials.
+### A) Backtest win rate (historical estimate)
+```bash
+python main.py --mode backtest --bars 800 --payout 0.82
+```
 
----
+### B) Journal-based performance report (actual bot outcomes)
+```bash
+python main.py --mode report --source all
+```
 
-## 📈 How Advanced Is This Bot?
+Or filter:
+```bash
+python main.py --mode report --source live_demo
+python main.py --mode report --source backtest
+```
 
-Implements the strategies explained in all top ICT resources, including:
-  - Market/Session model filtering
-  - Fair Value Gaps (FVG) and liquidity sweeps
-  - Order block logic
-  - 3-trade-per-day rule (no overtrading)
-  - Clean modular architecture
+Report includes:
+- trades, wins, losses
+- win rate %
+- profit factor
+- ROI %
+- max drawdown %
+- start/end balance
 
-Extremely easy to expand—just plug in more modules in `main.py` or update configs.
+All trades are saved in `trade_journal.csv` for audit/compliance review.
 
 ---
 
-**Want to go even higher? Deploy to the cloud! Ask for instructions on Replit/VPS/AWS/Heroku for no-pause 24/7 run.**
+## 🔧 Recommended `.env` for safer operation
+
+```env
+DEMO_MODE=True
+RISK_PERCENTAGE=1.0
+MAX_TRADES_PER_DAY=3
+MAX_DAILY_DRAWDOWN_PCT=3
+TRADE_COOLDOWN_MINUTES=5
+
+ENABLE_AI_FILTER=True
+AI_MODEL_PATH=trading_brain.pkl
+TRADE_MEMORY_PATH=trade_memory.csv
+TRADE_JOURNAL_PATH=trade_journal.csv
+AI_MIN_BUY_PROB=0.60
+AI_MAX_SELL_PROB=0.40
+AI_RETRAIN_EVERY_N_TRADES=25
+AI_MIN_RETRAIN_ROWS=40
+```
 
 ---
 
-**You're now truly at PRO level. Enjoy – and trade safe! 🏆**
\ No newline at end of file
+## ⚠️ Reality check
+
+No model can guarantee no mistakes. For company use, you should judge quality by:
+1. Out-of-sample accuracy from training,
+2. Live-demo journal metrics over enough trades,
+3. Controlled drawdown and consistent process.
