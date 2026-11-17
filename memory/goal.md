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

## Bots in the account (read this, daily reviewer)
- Scalper (scalper.py): intraday, flat before the close. Log: memory/scalper_trades.jsonl. Live stats since 2026-11-09.
- Swing bot (swing.py, swing.yml), since 2026-11-17, Ben chose "same account, small size": IBS on SMH - at ~15:55 ET buy when SMH is in the bottom 20% of its day range, sell after the bounce. 10% of the account, no leverage, -6% GTC stop at the broker, max 5 days. HOLDS OVERNIGHT by design (the scalper leaves SMH alone). Log: memory/swing_trades.jsonl, state memory/swing_state.json. Judge it with the same 4 criteria, separately (it trades ~1x/week, so criterion 2 takes longer - count weeks, not 60 trades).
- Learning: nightly backtest report has "Lessons" (FAVOR/AVOID conditions on BOTH halves) and "Live trades, judged by the same conditions". Strategy lab: memory/research/daily_*.md (research.yml, Fridays). Check swing.yml runs too.
- Ben's long-term wish: grow a small real account (e.g. $100). Only after the criteria hold; be honest that big multiples take years.
