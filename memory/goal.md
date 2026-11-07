# Goal: a scalper that is genuinely profitable (paper first)

The daily review checks these every trading day. "Profitable" = ALL of them true:

1. Backtest, last 60 days, on the unseen TEST part: profit > 0 and profit factor >= 1.3
2. Live paper trading for at least 4 weeks AND at least 60 closed trades
3. Live paper: net profit > 0, profit factor >= 1.3, max drawdown < 5% of the account
4. Live results roughly match the backtest (live profit factor not far below backtest)

Rules for improving:
- Change ONE thing at a time, and only if it is profitable on days the tuning never saw.
- Never loosen safety: paper only, stops at the broker, flat before the close, daily loss stop.
- Log every change and why in memory/changelog.md.

Only when all 4 are true for 2 weeks in a row: tell Ben it's reasonable to *consider* small real money.
