# Scalper changelog

- 2026-11-06: switched to selective mode (3x volume, first 2h, 2.4% stop / 4.8% target, 0.33x size). Reason: 60-day backtest, old settings -$63k; selective ~break-even.
- 2026-11-06: fixes: stop orders retried if rejected, runs always use newest code, trade logs merge cleanly.
- 2026-11-09: live day 1: 0 trades (3x-volume filter found no setup in the first 2h), account $98,381, flat. Added P/L-by-symbol/time breakdown to the backtest (report only). Finding: NVDA lost on both train (6 trades, PF 0.07) and test (5 trades, PF 0). NOT dropped yet: only 11 trades and the test days were already looked at; re-check on new days - drop NVDA if it keeps losing. Goal: 1/4 met (backtest test PF 2.21, but train PF 0.7 = fragile); 2-4 need 4 weeks/60 live trades.
