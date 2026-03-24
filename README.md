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
```bash
python -m venv venv
source venv/bin/activate            # On macOS/Linux
# OR: venv\\Scripts\\activate       # On Windows
```
## 2) Train AI model

**3. Install Requirements**
```bash
pip install -r requirements.txt
python train_ai.py --bars 2000 --horizon 1 --split-ratio 0.8
```

**4. Configure Your `.env` File**
Copy `.env.example` to `.env` and fill in your Pocket Option email/password.
## 3) Run bot

```bash
cp .env.example .env
python main.py --mode run
```
Edit `.env` to set your credentials, risk, and config.

---

## 🚀 Run the Bot
## 4) Backtest and reporting

```bash
python main.py
python main.py --mode backtest --bars 800 --payout 0.82
python main.py --mode report --source all
```

_The bot will run, trading a maximum of **3 trades per day** with full ICT analysis. See logs for trade actions and current balance._

---

## ⚠️ Important Notes
## Key config options (`config.py` or env)

- **Profit Guarantees:** No trading system can guarantee an 85% win rate; real markets have risk. Test in demo mode first.
- **Codespaces & 24/7:** Codespaces pause after inactivity! For true 24/7, consider deploying your bot on a VPS or a cloud service.
- **API Automation:** The bot auto-fetches the API endpoint each session. Update your `.env` only for new credentials.
- `DEMO_MODE=True|False`
- `TRAINING_PHASE=month1|month2|sniper`
- `MAX_ACCOUNT_USE_PER_TRADE_PCT=25`
- `STOP_LOSS_STAKE_PCT=40`
- `ANALYSIS_TIMEOUT_MINUTES=10`
- `HIGHER_TIMEFRAMES=[1w,1d,4h,1h]` (set in code config)
- `STRATEGY_KNOWLEDGE_PATH=strategy_knowledge.json`
- `TRADE_JOURNAL_PATH=trade_journal.csv`

---
## Notes

## 📈 How Advanced Is This Bot?
- Direct Pocket Option account balance sync is supported via `PO_BALANCE_API` endpoint if you provide one.
- This repository still simulates order execution unless a live execution adapter is added.
- No trading system can guarantee zero mistakes.

Implements the strategies explained in all top ICT resources, including:
  - Market/Session model filtering
  - Fair Value Gaps (FVG) and liquidity sweeps
  - Order block logic
  - 3-trade-per-day rule (no overtrading)
  - Clean modular architecture

Extremely easy to expand—just plug in more modules in `main.py` or update configs.
## Pocket Option live integration settings

---
Add these env vars for live account wiring:

**Want to go even higher? Deploy to the cloud! Ask for instructions on Replit/VPS/AWS/Heroku for no-pause 24/7 run.**

---
```env
PO_BASE_URL=https://<your-adapter-host>
PO_API_TOKEN=<your-secure-token>
PO_ACCOUNT_MODE=live
PO_ORDER_DURATION_SEC=60
PO_POLL_INTERVAL_SEC=1.5
```

**You're now truly at PRO level. Enjoy – and trade safe! 🏆**
If `PO_BASE_URL` + `PO_API_TOKEN` are set, the bot uses `PocketOptionClient` for balance/order/result calls.