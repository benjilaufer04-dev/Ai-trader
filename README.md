# AI Trader: AI-driven stock agent on Alpaca (free to run)

On every cycle (default: once an hour while the US market is open) the agent does this:

1. **Gathers** your account, open positions, about 6 months of daily prices for the watchlist (turned into returns, moving averages, RSI, volatility and volume), the last 48h of news, and its own recent decisions.
2. **Asks the AI** (Google Gemini by default, which is free) to give a market view and its buy/sell decisions as structured JSON, each with a confidence score and a reason.
3. **Runs a risk check** in plain code that the AI cannot override. The check rejects or shrinks anything that breaks your limits.
4. **Places orders** through Alpaca. It buys by dollar amount (fractional shares) and sells by a fraction of the position.
5. **Logs everything** to `journal.db` and `agent.log`, so you can see what it did and why.

```
data ──► AI (decides) ──► risk.py (guards) ──► Alpaca (executes) ──► journal
```

## Setup (about 15 minutes)

1. **Alpaca paper account.** Sign up at https://alpaca.markets. Paper trading is free and needs only an email. In the dashboard, switch to **Paper**, then go to **API Keys** → Generate.
2. **Free Gemini API key.** Go to https://aistudio.google.com/apikey, sign in with Google, and click **Create API key**. No card is needed.
3. **Easiest way:** double-click `start-windows.bat` (or `start-mac.command` on Mac). The first time, it installs everything and opens `.env` for your keys. After that, double-clicking it starts the agent, and closing the window stops it.

   **Manual way:**
   ```bash
   cd ai-trader
   python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env        # then paste your keys into .env
   python -m pytest -q         # risk-rule tests, should say 7 passed
   ```
4. **Run:**
   ```bash
   python agent.py --once --dry-run   # one full cycle, nothing is sent
   python agent.py --once             # one real cycle (paper money)
   python agent.py                    # run continuously
   python agent.py --report           # what it decided recently and why
   ```

## Risk limits (in `.env`)

| Setting | Default | Meaning |
|---|---|---|
| `MAX_POSITION_PCT` | 0.10 | No single stock above 10% of the account |
| `MAX_ORDER_USD` | 2000 | Largest single buy |
| `MAX_TRADES_PER_DAY` | 6 | Stops overtrading |
| `DAILY_LOSS_LIMIT_PCT` | 0.03 | Down 3% on the day: no new buys (sells still allowed) |
| `MIN_CASH_RESERVE_PCT` | 0.20 | Always keep 20% in cash |
| `MIN_CONFIDENCE` | 0.65 | Ignore trades the AI itself isn't sure about |
| `MAX_POSITIONS` | 8 | Max different stocks held |

Other protections:
- Long only, so no shorting and no margin bets.
- It only trades symbols on the `WATCHLIST`.
- **Kill switch:** create an empty file named `STOP` in the folder and all trading halts on the next cycle. Delete the file to resume.
- **Live trading lock:** real money needs `PAPER=false` *and* the exact phrase in `LIVE_TRADING_CONFIRM`. Without both, it refuses to start.

## Cost: $0

- Alpaca paper trading is free.
- Gemini's free tier covers about 7 decisions a day easily. Note that on the free tier Google may use the prompts (market data) to improve its products.
- It runs on your own computer, so there's no server bill. It only trades while the computer is on and `python agent.py` is running. Close it or turn the computer off and it simply stops, with nothing lost.
- Optional: set `AI_PROVIDER=anthropic` plus `ANTHROPIC_API_KEY` to use Claude instead (paid per use).

## Before you even think about live money

- Run it on paper for **at least 4–8 weeks**, then compare the result to just holding SPY over the same period. If it doesn't beat SPY after costs, the AI isn't adding value.
- Read `--report` regularly. Ask whether the reasoning makes sense or whether it's chasing headlines.
- Alpaca's live accounts for non-US residents depend on country. Check that Israel is supported on their site. If it isn't, the same design works with Interactive Brokers by swapping the order and account calls in `agent.py`.
- LLMs don't have a proven edge in short-term trading. Treat this as an experiment, and only ever go live with money you can afford to lose.

## Files

| File | Role |
|---|---|
| `agent.py` | Main loop, broker calls, CLI |
| `brain.py` | AI prompt and structured output (Gemini or Claude) |
| `market_data.py` | Prices, indicators, news (Alpaca free IEX feed) |
| `risk.py` | Hard guardrails (pure code, unit-tested) |
| `journal.py` | SQLite audit log |
| `config.py` / `.env` | All settings |
