"""Export market data for offline research (uploaded as a GitHub Actions artifact, never committed).

Writes data/minute.csv.gz (1-min bars, regular hours, last N trading days) and
data/daily.csv.gz (daily bars since 2016) for a broad symbol list, so strategy ideas can be
tested quickly outside the nightly run. Read-only: no orders, no account access.

Usage: python dump_data.py --minute-days 120 --since 2016-01-01
"""
import argparse
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

MINUTE_SYMBOLS = "TQQQ,SQQQ,SOXL,SOXS,TSLA,NVDA,QQQ,SPY,IWM,AAPL,AMD,META,AMZN,MSFT,TNA,TZA,UPRO,SPXU"
DAILY_SYMBOLS = ("SPY,QQQ,IWM,DIA,TQQQ,SQQQ,SOXL,SOXS,UPRO,TNA,TLT,GLD,SLV,USO,XLE,XLF,XLK,XLV,XLY,XLP,"
                 "XLI,XLU,SMH,ARKK,EEM,EFA,HYG,UUP,TSLA,NVDA,AAPL,MSFT,AMZN,META,GOOGL,AMD,NFLX,COIN,MSTR,PLTR")


def fetch(symbols, timeframe, start, end):
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    c = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    out = []
    for s in symbols:
        try:
            df = c.get_stock_bars(StockBarsRequest(symbol_or_symbols=s, timeframe=timeframe, start=start,
                                                   end=end, feed=DataFeed.IEX,
                                                   adjustment=Adjustment.ALL)).df
        except Exception as e:  # one bad symbol must not kill the export
            print(f"skip {s}: {e}")
            continue
        if len(df):
            out.append(df.reset_index())
            print(f"{s}: {len(df)} bars")
    return pd.concat(out) if out else pd.DataFrame()


def main():
    from alpaca.data.timeframe import TimeFrame
    ap = argparse.ArgumentParser()
    ap.add_argument("--minute-days", type=int, default=120)
    ap.add_argument("--since", default="2016-01-01")
    a = ap.parse_args()
    os.makedirs("data", exist_ok=True)
    end = datetime.now(timezone.utc) - timedelta(minutes=16)
    d = fetch(DAILY_SYMBOLS.split(","), TimeFrame.Day, datetime.fromisoformat(a.since).replace(tzinfo=timezone.utc), end)
    d.to_csv("data/daily.csv.gz", index=False)
    if a.minute_days <= 0:
        print(f"daily rows {len(d)} (minute bars skipped)")
        return
    m = fetch(MINUTE_SYMBOLS.split(","), TimeFrame.Minute, end - timedelta(days=int(a.minute_days * 1.5) + 4), end)
    if len(m):
        ny = pd.to_datetime(m["timestamp"]).dt.tz_convert("America/New_York")
        m = m[(ny.dt.time >= pd.Timestamp("09:30").time()) & (ny.dt.time < pd.Timestamp("16:00").time())]
    m.to_csv("data/minute.csv.gz", index=False)
    print(f"daily rows {len(d)}, minute rows {len(m)}")


if __name__ == "__main__":
    main()
