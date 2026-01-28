# Pocket Option ICT "5 Star" Pro Bot

_A fully automated, high-class ICT trading bot for Pocket Option, entirely configurable and built for precision trading._  
**Author**: leinsz-vaios

---

## ⚡ Features
- ✓ Fair Value Gap (FVG), Order Block, and Market Structure logic
- ✓ Automatic Pocket Option API endpoint freshness check
- ✓ Per-account API key/password – no hardcoded secrets!
- ✓ Up to **3 trades/day**, all risk-managed
- ✓ Full session filtering (London/NY only)
- ✓ Runs in demo or real mode (set in `.env`)
- ✓ Simple, clear logs for every action

---

## 🛠️ Setup Instructions

**1. Get the Code**
```bash
git clone https://github.com/leinsz-vaios/ict-trading-bot.git
cd ict-trading-bot
```

**2. Configure Python Environment**
```bash
python -m venv venv
source venv/bin/activate            # On macOS/Linux
# OR: venv\\Scripts\\activate       # On Windows
```

**3. Install Requirements**
```bash
pip install -r requirements.txt
```

**4. Configure Your `.env` File**
Copy `.env.example` to `.env` and fill in your Pocket Option email/password.
```bash
cp .env.example .env
```
Edit `.env` to set your credentials, risk, and config.

---

## 🚀 Run the Bot

```bash
python main.py
```

_The bot will run, trading a maximum of **3 trades per day** with full ICT analysis. See logs for trade actions and current balance._

---

## ⚠️ Important Notes

- **Profit Guarantees:** No trading system can guarantee an 85% win rate; real markets have risk. Test in demo mode first.
- **Codespaces & 24/7:** Codespaces pause after inactivity! For true 24/7, consider deploying your bot on a VPS or a cloud service.
- **API Automation:** The bot auto-fetches the API endpoint each session. Update your `.env` only for new credentials.

---

## 📈 How Advanced Is This Bot?

Implements the strategies explained in all top ICT resources, including:
  - Market/Session model filtering
  - Fair Value Gaps (FVG) and liquidity sweeps
  - Order block logic
  - 3-trade-per-day rule (no overtrading)
  - Clean modular architecture

Extremely easy to expand—just plug in more modules in `main.py` or update configs.

---

**Want to go even higher? Deploy to the cloud! Ask for instructions on Replit/VPS/AWS/Heroku for no-pause 24/7 run.**

---

**You're now truly at PRO level. Enjoy – and trade safe! 🏆**