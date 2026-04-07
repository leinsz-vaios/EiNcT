# Pocket Option ICT "5 Star" Pro Bot
# ICT Trading Bot (Spec-Aligned)

_A fully automated, high-class ICT trading bot for Pocket Option, entirely configurable and built for precision trading._  
**Author**: leinsz-vaios
This bot now supports:
- Multi-timeframe analysis (`1w`, `1d`, `4h`, `1h`) + execution on `1m`
- Strategy ingestion from docs/videos into JSON knowledge
- Mandatory daily strategy review before trading
- Training phases (`month1`, `month2`, `sniper`)
- Rich trade journaling + reward score
- AI filter + retraining from memory

---
## 1) Ingest strategy docs + videos into JSON

## ⚡ Features
- ✓ Fair Value Gap (FVG), Order Block, and Market Structure logic
- ✓ Automatic Pocket Option API endpoint freshness check
- ✓ Per-account API key/password – no hardcoded secrets!
- ✓ Up to **3 trades/day**, all risk-managed
- ✓ Full session filtering (London/NY only)
- ✓ Runs in demo or real mode (set in `.env`)
- ✓ Simple, clear logs for every action
Put your files in `strategy_docs/` (including `.docx`) then run:

---

## 🛠️ Setup Instructions

**1. Get the Code**
```bash
git clone https://github.com/leinsz-vaios/ict-trading-bot.git
cd ict-trading-bot
python ingest_strategy.py \
  --docs-dir strategy_docs \
  --out strategy_knowledge.json \
  --video "https://youtu.be/aWFSvqxEfg8?si=z0M4GkqXA8taXHli" \
  --video "https://www.youtube.com/watch?v=Ue1DfHoYo48" \
  --video "https://www.youtube.com/watch?v=OhE__u454wo" \
  --video "https://www.youtube.com/watch?v=0fTcijWBlzA"
```

**2. Configure Python Environment**
## 2) Train AI model

```bash
python -m venv venv
source venv/bin/activate            # On macOS/Linux
# OR: venv\\Scripts\\activate       # On Windows
python train_ai.py --bars 2000 --horizon 1 --split-ratio 0.8
```

**3. Install Requirements**
## 3) Run bot

```bash
pip install -r requirements.txt
python main.py --mode run
```

**4. Configure Your `.env` File**
Copy `.env.example` to `.env` and fill in your Pocket Option email/password.
## 4) Backtest and reporting

```bash
cp .env.example .env
python main.py --mode backtest --bars 800 --payout 0.82
python main.py --mode report --source all
```
Edit `.env` to set your credentials, risk, and config.

---
## Key config options (`config.py` or env)

## 🚀 Run the Bot
- `DEMO_MODE=True|False`
- `TRAINING_PHASE=month1|month2|sniper`
- `MAX_ACCOUNT_USE_PER_TRADE_PCT=25`
- `STOP_LOSS_STAKE_PCT=40`
- `ANALYSIS_TIMEOUT_MINUTES=10`
- `MARKET_DATA_EXCHANGES=binance,kraken,coinbase`
- `HIGHER_TIMEFRAMES=[1w,1d,4h,1h]` (set in code config)
- `STRATEGY_KNOWLEDGE_PATH=strategy_knowledge.json`
- `TRADE_JOURNAL_PATH=trade_journal.csv`

```bash
python main.py
## Notes

- Direct Pocket Option account balance sync is supported via `PO_BALANCE_API` endpoint if you provide one.
- This repository still simulates order execution unless a live execution adapter is added.
- No trading system can guarantee zero mistakes.


## Pocket Option live integration settings

Add these env vars for live account wiring:

```env
PO_BASE_URL=https://<your-adapter-host>
PO_API_TOKEN=<your-secure-token>
PO_ACCOUNT_MODE=live
PO_ORDER_DURATION_SEC=60
PO_POLL_INTERVAL_SEC=1.5
```

_The bot will run, trading a maximum of **3 trades per day** with full ICT analysis. See logs for trade actions and current balance._
If `PO_BASE_URL` + `PO_API_TOKEN` are set, the bot uses `PocketOptionClient` for balance/order/result calls.

## Railway deployment notes (important)

---
If Railway logs show:

## ⚠️ Important Notes
```text
python: can't open file '/app/main.py': [Errno 2] No such file or directory
```

your persistent volume is likely mounted over `/app`, which hides the code that Railway cloned during build.

- **Profit Guarantees:** No trading system can guarantee an 85% win rate; real markets have risk. Test in demo mode first.
- **Codespaces & 24/7:** Codespaces pause after inactivity! For true 24/7, consider deploying your bot on a VPS or a cloud service.
- **API Automation:** The bot auto-fetches the API endpoint each session. Update your `.env` only for new credentials.
Use this setup instead:

---
- **Do not mount a volume at `/app`**
- Mount volume at something like `/data`
- Keep your start command as `python main.py` (working directory remains `/app`)
- Point writable artifacts to `/data` via env vars, for example:

## 📈 How Advanced Is This Bot?
```env
AI_MODEL_PATH=/data/trading_brain.pkl
TRADE_MEMORY_PATH=/data/trade_memory.csv
TRADE_JOURNAL_PATH=/data/trade_journal.csv
STRATEGY_REVIEW_STATE_PATH=/data/strategy_review_state.json
```

Implements the strategies explained in all top ICT resources, including:
  - Market/Session model filtering
  - Fair Value Gaps (FVG) and liquidity sweeps
  - Order block logic
  - 3-trade-per-day rule (no overtrading)
  - Clean modular architecture
If build fails with:

Extremely easy to expand—just plug in more modules in `main.py` or update configs.
```text
ERROR: failed to solve: secret PO_EMAIL: not found
```

---
that is a Railway configuration issue (missing Build Secret reference), not a Python code issue.

**Want to go even higher? Deploy to the cloud! Ask for instructions on Replit/VPS/AWS/Heroku for no-pause 24/7 run.**
Fix it by either:

---
- removing `PO_EMAIL` from **Build Secrets** / Docker secret references, or
- defining `PO_EMAIL` in Railway variables if you intentionally use it.

**You're now truly at PRO level. Enjoy – and trade safe! 🏆**
`PO_EMAIL` / `PO_PASSWORD` are optional in this repo unless your own custom adapter requires them; the HTTP adapter path uses `PO_BASE_URL` + `PO_API_TOKEN`.