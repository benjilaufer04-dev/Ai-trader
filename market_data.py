"""Price history, technical indicators and news for the watchlist."""
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame


def rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-9)
    return float(100 - 100 / (1 + rs.iloc[-1]))


def summarize(df: pd.DataFrame) -> dict:
    """Turn ~6 months of daily bars into a compact snapshot the LLM can read."""
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    last = float(c.iloc[-1])

    def ret(n):
        return round((last / float(c.iloc[-n - 1]) - 1) * 100, 2) if len(c) > n else None

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    sma20, sma50 = c.rolling(20).mean().iloc[-1], c.rolling(50).mean().iloc[-1]
    return {
        "price": round(last, 2),
        "ret_1d_%": ret(1), "ret_5d_%": ret(5), "ret_20d_%": ret(20), "ret_60d_%": ret(60),
        "vs_sma20_%": round(float(last / sma20 - 1) * 100, 2) if pd.notna(sma20) else None,
        "vs_sma50_%": round(float(last / sma50 - 1) * 100, 2) if pd.notna(sma50) else None,
        "rsi14": round(rsi(c), 1),
        "atr14_%": round(float(tr.rolling(14).mean().iloc[-1]) / last * 100, 2),
        "volume_vs_20d_avg": round(float(v.iloc[-1] / v.rolling(20).mean().iloc[-1]), 2),
        "high_52w_dist_%": round((last / float(h.max()) - 1) * 100, 2),
    }


class MarketData:
    def __init__(self, key, secret):
        self.bars = StockHistoricalDataClient(key, secret)
        self.news = NewsClient(key, secret)

    def snapshots(self, symbols: list) -> dict:
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=datetime.now(timezone.utc) - timedelta(days=260),
            feed=DataFeed.IEX,  # free data feed
        )
        df = self.bars.get_stock_bars(req).df
        out = {}
        for sym in symbols:
            try:
                sdf = df.xs(sym, level="symbol")
                if len(sdf) >= 30:
                    out[sym] = summarize(sdf)
            except KeyError:
                pass
        return out

    def headlines(self, symbols: list, hours: int = 48, limit: int = 40) -> list:
        req = NewsRequest(
            symbols=",".join(symbols),
            start=datetime.now(timezone.utc) - timedelta(hours=hours),
            limit=limit,
            sort="desc",
        )
        items = self.news.get_news(req)
        news = items.data.get("news", []) if hasattr(items, "data") else items.news
        return [
            {
                "time": n.created_at.strftime("%m-%d %H:%M"),
                "symbols": [s for s in n.symbols if s in symbols],
                "headline": n.headline,
                "summary": (n.summary or "")[:280],
            }
            for n in news
        ]
