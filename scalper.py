"""Scalper: rule-based, 1-minute momentum scalping with big leveraged positions. PAPER ONLY.

No AI in the loop - fixed rules, checked every ~20 seconds:

ENTRY (long only; inverse ETFs like SQQQ/SOXS are how it profits from a fall)
  on the last COMPLETE 1-minute bar:
  - price above session VWAP            (buyers in control today)
  - EMA9 above EMA21                    (short-term trend up)
  - close breaks the high of the 5 bars before it (breakout)
  - green bar on volume >= 1.3x the 20-bar average (real push, not noise)
EXIT (whichever comes first)
  - stop-loss   -0.5% from entry
  - take-profit +1.0% (2x the risk)
  - at +0.5% the stop moves to breakeven; at +0.8% it locks in +0.4%
  - momentum dies: EMA9 < EMA21 and price < VWAP
  - time stop: 20 min without reaching +0.3%
  - everything is closed 10 min before the close (never holds overnight)
SIZE
  - each position = POSITION_X x account equity (default 1.5x, i.e. borrowed money)
  - max 2 positions at once (up to ~3x leverage; Alpaca paper allows 4x intraday)
  - account down DAILY_STOP_PCT (4%) on the day -> no new trades today
  - 10 min cooldown on a symbol after closing it

Usage:  python scalper.py                 run until the close (or --minutes N)
        python scalper.py --dry-run       print what it would do, no orders
"""
import argparse
import json
import logging
import math
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("scalper")


def _f(n, d):
    return float(os.getenv(n, d))


CFG = {
    "symbols": [s.strip().upper() for s in os.getenv(
        "SCALP_SYMBOLS", "TQQQ,SQQQ,SOXL,SOXS,TSLA,NVDA").split(",") if s.strip()],
    "position_x": _f("POSITION_X", 1.5),        # each position = 1.5x equity
    "max_positions": int(_f("MAX_POSITIONS", 2)),
    "stop_pct": _f("STOP_PCT", 0.005),
    "target_pct": _f("TARGET_PCT", 0.010),
    "breakeven_at": _f("BREAKEVEN_AT", 0.005),
    "lock_at": _f("LOCK_AT", 0.008), "lock_pct": _f("LOCK_PCT", 0.004),
    "time_stop_min": _f("TIME_STOP_MIN", 20), "time_stop_needs": _f("TIME_STOP_NEEDS", 0.003),
    "vol_mult": _f("VOL_MULT", 1.3),
    "daily_stop_pct": _f("DAILY_STOP_PCT", 0.04),
    "cooldown_min": _f("COOLDOWN_MIN", 10),
    "skip_open_min": _f("SKIP_OPEN_MIN", 5), "flatten_before_close_min": _f("FLATTEN_MIN", 10),
    "no_new_before_close_min": _f("NO_NEW_MIN", 20),
    "loop_seconds": _f("LOOP_SECONDS", 20),
}
TRADES = os.path.join(os.getenv("MEMORY_DIR", "memory"), "scalper_trades.jsonl")


# ---------- pure logic (tested in test_scalper.py) ----------

def indicators(df):
    """df: today's 1-min bars (open/high/low/close/volume), oldest first."""
    c = df["close"]
    tp = (df["high"] + df["low"] + c) / 3
    return {
        "ema9": c.ewm(span=9, adjust=False).mean(),
        "ema21": c.ewm(span=21, adjust=False).mean(),
        "vwap": (tp * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, math.nan),
        "vol_avg": df["volume"].rolling(20).mean(),
    }


def entry_signal(df, cfg=CFG):
    """Returns a strength score (>0 = buy) for the last complete bar, else 0."""
    if len(df) < 25:
        return 0
    ind = indicators(df)
    i = len(df) - 1
    close, open_ = df["close"].iloc[i], df["open"].iloc[i]
    vol, vavg = df["volume"].iloc[i], ind["vol_avg"].iloc[i - 1]
    prior_high = df["high"].iloc[i - 5:i].max()
    ok = (close > ind["vwap"].iloc[i] and ind["ema9"].iloc[i] > ind["ema21"].iloc[i]
          and close > prior_high and close > open_ and vavg > 0 and vol >= cfg["vol_mult"] * vavg)
    return float(vol / vavg) if ok else 0


def momentum_dead(df):
    if len(df) < 22:
        return False
    ind = indicators(df)
    return bool(ind["ema9"].iloc[-1] < ind["ema21"].iloc[-1] and df["close"].iloc[-1] < ind["vwap"].iloc[-1])


def manage(trade, price, now, cfg=CFG):
    """trade: {"entry","stop","target","opened"(datetime),"best"}. Updates stop in place.
    Returns an exit reason or None."""
    trade["best"] = max(trade.get("best", price), price)
    gain_best = trade["best"] / trade["entry"] - 1
    if gain_best >= cfg["lock_at"]:
        trade["stop"] = max(trade["stop"], trade["entry"] * (1 + cfg["lock_pct"]))
    elif gain_best >= cfg["breakeven_at"]:
        trade["stop"] = max(trade["stop"], trade["entry"])
    if price <= trade["stop"]:
        return f"stop {trade['stop']:.2f}"
    if price >= trade["target"]:
        return f"target {trade['target']:.2f}"
    age = (now - trade["opened"]).total_seconds() / 60
    if age >= cfg["time_stop_min"] and gain_best < cfg["time_stop_needs"]:
        return f"time stop ({age:.0f} min, no follow-through)"
    return None


def size(equity, buying_power, price, n_open, cfg=CFG):
    usd = min(equity * cfg["position_x"], buying_power * 0.95)
    return int(usd // price) if price > 0 and n_open < cfg["max_positions"] else 0


# ---------- live loop ----------

class Scalper:
    def __init__(self, dry_run=False):
        key, sec = os.getenv("ALPACA_API_KEY", ""), os.getenv("ALPACA_SECRET_KEY", "")
        if not key or not sec:
            raise SystemExit("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY")
        if os.getenv("PAPER", "true").lower() == "false":
            raise SystemExit("The scalper is a paper-trading experiment only. Refusing to run live.")
        self.t = TradingClient(key, sec, paper=True)
        self.d = StockHistoricalDataClient(key, sec)
        self.dry = dry_run
        self.trades, self.cooldown = {}, {}
        self.day_start_equity = float(self.t.get_account().last_equity)

    def bars(self, since):
        df = self.d.get_stock_bars(StockBarsRequest(symbol_or_symbols=CFG["symbols"], timeframe=TimeFrame.Minute,
                                                    start=since, feed=DataFeed.IEX)).df
        out = {}
        if df is None or df.empty:
            return out
        now_min = pd.Timestamp.now(tz="UTC").floor("min")
        for s in CFG["symbols"]:
            if s in df.index.get_level_values(0):
                x = df.xs(s).sort_index()
                out[s] = x[x.index < now_min]  # complete bars only
        return out

    def prices(self, syms):
        if not syms:
            return {}
        r = self.d.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=list(syms), feed=DataFeed.IEX))
        return {s: float(v.price) for s, v in r.items()}

    def order(self, sym, qty, side):
        if self.dry:
            log.info("  [dry-run] %s %s %s", side, qty, sym)
            return None
        o = self.t.submit_order(MarketOrderRequest(symbol=sym, qty=qty, side=side, time_in_force=TimeInForce.DAY))
        for _ in range(10):
            time.sleep(0.5)
            o = self.t.get_order_by_id(o.id)
            if o.filled_avg_price:
                return float(o.filled_avg_price)
        return None

    def record(self, row):
        if self.dry:
            return
        os.makedirs(os.path.dirname(TRADES), exist_ok=True)
        with open(TRADES, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def close(self, sym, reason, price):
        tr = self.trades.pop(sym, None)
        try:
            if not self.dry:
                self.t.close_position(sym)
        except Exception as e:
            log.error("  CLOSE FAILED %s: %s", sym, e)
            return
        self.cooldown[sym] = datetime.now(timezone.utc) + timedelta(minutes=CFG["cooldown_min"])
        if tr:
            pnl = (price - tr["entry"]) * tr["qty"]
            log.info("EXIT  %-5s %d @ ~%.2f  P/L %+.2f (%+.2f%%)  [%s]", sym, tr["qty"], price, pnl,
                     (price / tr["entry"] - 1) * 100, reason)
            self.record({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "symbol": sym,
                         "side": "sell", "qty": tr["qty"], "entry": tr["entry"], "exit": price,
                         "pnl": round(pnl, 2), "reason": reason})
        else:
            log.info("EXIT  %-5s (leftover position) [%s]", sym, reason)

    def adopt_positions(self):
        """On start: keep tracking our own symbols, close anything else (leftovers from other bots)."""
        for p in self.t.get_all_positions():
            sym, entry = p.symbol, float(p.avg_entry_price)
            if sym in CFG["symbols"] and float(p.qty) > 0:
                self.trades[sym] = {"entry": entry, "qty": int(float(p.qty)),
                                    "stop": entry * (1 - CFG["stop_pct"]), "target": entry * (1 + CFG["target_pct"]),
                                    "opened": datetime.now(timezone.utc), "best": entry}
                log.info("ADOPT %s %s @ %.2f", sym, p.qty, entry)
            else:
                self.close(sym, "not a scalping symbol", float(p.current_price))

    def run(self, minutes=None):
        clock = self.t.get_clock()
        while not clock.is_open:
            wait = (clock.next_open - clock.timestamp).total_seconds()
            if wait > 3 * 3600:
                log.info("Market closed, next open %s - exiting.", clock.next_open)
                return
            log.info("Waiting %.0f min for the open...", wait / 60)
            time.sleep(min(wait + 5, 600))
            clock = self.t.get_clock()
        close_at = clock.next_close
        stop_at = datetime.now(timezone.utc) + timedelta(minutes=minutes) if minutes else close_at
        session_start = (close_at - timedelta(hours=6, minutes=30))
        log.info("Scalping %s until %s | position %.1fx equity, max %d at once",
                 ",".join(CFG["symbols"]), min(stop_at, close_at), CFG["position_x"], CFG["max_positions"])
        self.adopt_positions()

        while True:
            now = datetime.now(timezone.utc)
            to_close = (close_at - now).total_seconds() / 60
            if to_close <= CFG["flatten_before_close_min"]:
                for s in list(self.trades):
                    self.close(s, "end of day - flat overnight", self.prices([s]).get(s, self.trades[s]["entry"]))
                log.info("Flat for the day.")
                break
            if now >= stop_at:
                # Hand over flat: never leave leveraged trades unwatched between GitHub runs.
                for s in list(self.trades):
                    self.close(s, "run time limit - handing over flat", self.prices([s]).get(s, self.trades[s]["entry"]))
                log.info("Run time limit reached, next run takes over.")
                break
            try:
                self.step(now, session_start, to_close)
            except Exception as e:
                log.exception("loop error: %s", e)
            time.sleep(CFG["loop_seconds"])
        self.summary()

    def step(self, now, session_start, to_close):
        acct = self.t.get_account()
        equity, bp = float(acct.equity), float(acct.buying_power)
        # 1) manage open trades every loop
        px = self.prices(self.trades)
        bars = None
        for s, tr in list(self.trades.items()):
            p = px.get(s)
            if not p:
                continue
            why = manage(tr, p, now)
            if not why:
                bars = bars or self.bars(session_start)
                if s in bars and momentum_dead(bars[s]):
                    why = "momentum faded (EMA9<EMA21, below VWAP)"
            if why:
                self.close(s, why, p)
        # 2) look for new entries
        day_pnl = equity / self.day_start_equity - 1
        if day_pnl <= -CFG["daily_stop_pct"]:
            return
        if (now - session_start).total_seconds() / 60 < CFG["skip_open_min"] or to_close < CFG["no_new_before_close_min"]:
            return
        if len(self.trades) >= CFG["max_positions"]:
            return
        bars = bars or self.bars(session_start)
        cands = sorted(((entry_signal(df), s) for s, df in bars.items()
                        if s not in self.trades and self.cooldown.get(s, now) <= now), reverse=True)
        for score, s in cands:
            if score <= 0 or len(self.trades) >= CFG["max_positions"]:
                break
            price = float(bars[s]["close"].iloc[-1])
            qty = size(equity, bp, price, len(self.trades))
            if qty < 1:
                break
            fill = self.order(s, qty, OrderSide.BUY) or price
            self.trades[s] = {"entry": fill, "qty": qty, "stop": fill * (1 - CFG["stop_pct"]),
                              "target": fill * (1 + CFG["target_pct"]), "opened": now, "best": fill}
            bp -= qty * fill
            log.info("ENTRY %-5s %d @ %.2f (= $%s, %.1fx equity) vol x%.1f | stop %.2f target %.2f",
                     s, qty, fill, f"{qty*fill:,.0f}", qty * fill / equity, score,
                     self.trades[s]["stop"], self.trades[s]["target"])
            self.record({"ts": now.isoformat(timespec="seconds"), "symbol": s, "side": "buy", "qty": qty,
                         "entry": fill, "signal_volume_x": round(score, 2)})

    def summary(self):
        a = self.t.get_account()
        eq, last = float(a.equity), float(a.last_equity)
        today = datetime.now(timezone.utc).date().isoformat()
        rows = []
        if os.path.exists(TRADES):
            rows = [json.loads(l) for l in open(TRADES) if l.startswith('{"ts": "' + today)]
        exits = [r for r in rows if r["side"] == "sell" and "pnl" in r]
        wins = [r for r in exits if r["pnl"] > 0]
        log.info("SUMMARY today: account %+.2f (%+.2f%%) | %d trades closed, %d wins, %d losses, P/L %+.2f",
                 eq - last, (eq / last - 1) * 100, len(exits), len(wins), len(exits) - len(wins),
                 sum(r["pnl"] for r in exits))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, help="stop after N minutes (GitHub jobs max out at 6h)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    Scalper(dry_run=args.dry_run).run(args.minutes)
