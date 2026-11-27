"""Swing bot: 'buy the weak close' (IBS) on SMH, small size, holds overnight. PAPER ONLY.

The pattern (found by research.py, checked on unseen years, a placebo luck test, and with real
15:55 prices): when a fund closes near the LOW of its day, it tends to bounce the next day.

Once per day, ~5 minutes before the close:
  IBS = (price - today's low) / (today's high - today's low)    (0 = at the low, 1 = at the high)
  - not holding and IBS < SWING_IBS (0.2)  -> buy SWING_PCT (10%) of the account, no leverage
  - holding and IBS < SWING_IBS again      -> keep holding one more day (max SWING_MAX_DAYS)
  - holding otherwise                      -> sell
Safety: a protective stop order at the broker (SWING_STOP_PCT, -6%) is always in place overnight;
no leverage; PAPER lock; skips the day if it starts too late to trade before the close.
The scalper leaves SWING_SYMBOLS alone (same paper account).

Usage: python swing.py [--dry-run] [--now]   (--now: decide immediately, for testing)
"""
import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("swing")

CFG = {
    "symbols": [s.strip().upper() for s in os.getenv("SWING_SYMBOLS", "SMH").split(",") if s.strip()],
    "ibs": float(os.getenv("SWING_IBS", 0.2)),
    "pct": float(os.getenv("SWING_PCT", 0.10)),          # of account equity per position
    "stop_pct": float(os.getenv("SWING_STOP_PCT", 0.06)),
    "max_days": int(os.getenv("SWING_MAX_DAYS", 5)),
    "decide_min_before_close": float(os.getenv("SWING_DECIDE_MIN", 5)),
    "too_late_min": float(os.getenv("SWING_TOO_LATE_MIN", 2)),
    "max_wait_min": float(os.getenv("SWING_MAX_WAIT_MIN", 75)),
}
TRADES = os.path.join(os.getenv("MEMORY_DIR", "memory"), "swing_trades.jsonl")
STATE = os.path.join(os.getenv("MEMORY_DIR", "memory"), "swing_state.json")


# ---------- pure logic (tested in test_swing.py) ----------

def ibs(high, low, price):
    return (price - low) / (high - low) if high > low else 0.5


def decide(holding, ibs_now, days_held, cfg=CFG):
    """'buy', 'sell', 'hold' or 'wait'."""
    weak = ibs_now < cfg["ibs"]
    if not holding:
        return "buy" if weak else "wait"
    if weak and days_held < cfg["max_days"]:
        return "hold"
    return "sell"


def qty_for(equity, cash, price, cfg=CFG):
    """Whole shares worth SWING_PCT of equity, never more than the cash we have (no leverage)."""
    usd = min(equity * cfg["pct"], max(cash, 0) * 0.95)
    return int(usd // price) if price > 0 else 0


def minutes_plan(min_to_close, cfg=CFG):
    """What to do given minutes until the close: 'skip' (wrong run / too far), 'late' (can't trade
    safely any more), or the number of minutes to wait before deciding."""
    if min_to_close > cfg["max_wait_min"]:
        return "skip"
    if min_to_close < cfg["too_late_min"]:
        return "late"
    return max(0.0, min_to_close - cfg["decide_min_before_close"])


# ---------- live ----------

class Swing:
    def __init__(self, dry=False):
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient
        key, sec = os.getenv("ALPACA_API_KEY", ""), os.getenv("ALPACA_SECRET_KEY", "")
        if not key or not sec:
            raise SystemExit("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY")
        if os.getenv("PAPER", "true").lower() == "false":
            raise SystemExit("The swing bot is a paper-trading experiment only. Refusing to run live.")
        self.t = TradingClient(key, sec, paper=True)
        self.d = StockHistoricalDataClient(key, sec)
        self.dry = dry
        self.state = json.load(open(STATE)) if os.path.exists(STATE) else {}

    def save(self):
        os.makedirs(os.path.dirname(STATE) or ".", exist_ok=True)
        json.dump(self.state, open(STATE, "w"), indent=1)

    def record(self, row):
        os.makedirs(os.path.dirname(TRADES) or ".", exist_ok=True)
        with open(TRADES, "a") as f:
            f.write(json.dumps(row) + "\n")

    def today_bars(self, sym, session_start):
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        df = self.d.get_stock_bars(StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Minute,
                                                    start=session_start, feed=DataFeed.IEX)).df
        return df.xs(sym).sort_index() if df is not None and len(df) else None

    def last_price(self, sym):
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockLatestTradeRequest
        return float(self.d.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=sym,
                                                                           feed=DataFeed.IEX))[sym].price)

    def cancel_orders(self, sym):
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        for o in self.t.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[sym])):
            self.t.cancel_order_by_id(o.id)
        time.sleep(2)

    def wait_fill(self, oid, tries=30):
        for _ in range(tries):
            o = self.t.get_order_by_id(oid)
            if str(o.status).lower().endswith("filled") and float(o.filled_qty or 0) > 0 \
                    and not str(o.status).lower().endswith("partially_filled"):
                return float(o.filled_avg_price), int(float(o.filled_qty))
            time.sleep(2)
        o = self.t.get_order_by_id(oid)
        return (float(o.filled_avg_price) if o.filled_avg_price else None), int(float(o.filled_qty or 0))

    def place_stop(self, sym, qty, entry):
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import StopOrderRequest
        stop = round(entry * (1 - CFG["stop_pct"]), 2)
        for attempt in range(3):
            try:
                o = self.t.submit_order(StopOrderRequest(symbol=sym, qty=qty, side=OrderSide.SELL,
                                                         time_in_force=TimeInForce.GTC, stop_price=stop))
                log.info("STOP  %s %d @ %.2f (GTC, at the broker)", sym, qty, stop)
                return o.id
            except Exception as e:
                log.error("stop order failed (try %d): %s", attempt + 1, str(e)[:200])
                time.sleep(10)
        return None

    def run(self, now_mode=False):
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        clock = self.t.get_clock()
        if not clock.is_open:
            log.info("Market closed - nothing to do.")
            return
        to_close = (clock.next_close - clock.timestamp).total_seconds() / 60
        plan = 0.0 if now_mode else minutes_plan(to_close)
        if plan == "skip":
            log.info("%.0f min to the close - not this run (the other scheduled run handles today).", to_close)
            return
        if plan == "late":
            log.warning("Started too late (%.1f min to the close) - no trade today.", to_close)
            return
        log.info("Waiting %.1f min, then deciding %s min before the close.", plan, CFG["decide_min_before_close"])
        time.sleep(plan * 60)
        # 9:30 ET today (not close - 6.5h: wrong on half days, would pull in pre-market bars)
        session_start = clock.next_close.astimezone(ZoneInfo("America/New_York")).replace(
            hour=9, minute=30, second=0, microsecond=0)
        acct = self.t.get_account()
        equity, cash = float(acct.equity), float(acct.cash)
        held = {p.symbol: p for p in self.t.get_all_positions()}
        today = datetime.now(timezone.utc).date().isoformat()
        # positions the stop already closed overnight: clean up state + log
        for sym in list(self.state):
            if sym not in held and sym in CFG["symbols"] and not self.dry:
                st = self.state.pop(sym)
                log.info("%s was sold by its stop order since the last run.", sym)
                self.record({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "symbol": sym,
                             "side": "sell", "qty": st.get("qty"), "entry": st.get("entry"),
                             "exit": round(st.get("entry", 0) * (1 - CFG["stop_pct"]), 2),
                             "pnl": round(-st.get("entry", 0) * CFG["stop_pct"] * (st.get("qty") or 0), 2),
                             "reason": "stop order (approx. price)"})
        for sym in CFG["symbols"]:
            bars = self.today_bars(sym, session_start)
            if bars is None or len(bars) < 300:
                log.warning("%s: not enough bars today (%s) - skipping.", sym, 0 if bars is None else len(bars))
                continue
            price = self.last_price(sym)
            hi, lo = max(bars["high"].max(), price), min(bars["low"].min(), price)
            v = ibs(hi, lo, price)
            st = self.state.get(sym, {})
            holding = sym in held and float(held[sym].qty) > 0
            days = st.get("days", 0) + (1 if holding and st.get("last_day") != today else 0)
            action = decide(holding, v, days)
            log.info("%s price %.2f | day high %.2f low %.2f | IBS %.2f (buy below %.2f) | holding %s (%d days) -> %s",
                     sym, price, hi, lo, v, CFG["ibs"], holding, days, action.upper())
            if self.dry:
                continue
            ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            if action == "buy":
                q = qty_for(equity, cash, price)
                if q < 1:
                    log.info("%s: size rounds to 0 shares - skip.", sym)
                    continue
                o = self.t.submit_order(MarketOrderRequest(symbol=sym, qty=q, side=OrderSide.BUY,
                                                           time_in_force=TimeInForce.DAY))
                fill, q = self.wait_fill(o.id)
                if not q:
                    log.error("%s: buy did not fill.", sym)
                    continue
                fill = fill or price
                self.place_stop(sym, q, fill)   # only after the full fill (wash-trade reject otherwise)
                self.state[sym] = {"entry": fill, "qty": q, "days": 0, "last_day": today, "opened": ts}
                self.record({"ts": ts, "symbol": sym, "side": "buy", "qty": q, "entry": fill, "ibs": round(v, 3)})
                log.info("BUY   %s %d @ %.2f (= $%s, %.0f%% of the account)", sym, q, fill, f"{q * fill:,.0f}",
                         q * fill / equity * 100)
            elif action == "hold":
                self.state[sym] = dict(st, days=days, last_day=today)
                log.info("HOLD  %s one more day (closed weak again).", sym)
            elif action == "sell":
                q = int(float(held[sym].qty))
                self.cancel_orders(sym)           # free the shares held by the stop order
                o = self.t.submit_order(MarketOrderRequest(symbol=sym, qty=q, side=OrderSide.SELL,
                                                           time_in_force=TimeInForce.DAY))
                fill, _ = self.wait_fill(o.id)
                fill = fill or price
                entry = st.get("entry") or float(held[sym].avg_entry_price)
                pnl = (fill - entry) * q
                self.record({"ts": ts, "symbol": sym, "side": "sell", "qty": q, "entry": entry, "exit": fill,
                             "pnl": round(pnl, 2), "days": days, "ibs": round(v, 3)})
                self.state.pop(sym, None)
                log.info("SELL  %s %d @ %.2f  P/L %+.2f (%+.2f%%) after %d day(s)", sym, q, fill, pnl,
                         (fill / entry - 1) * 100, days)
        if not self.dry:
            self.save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--now", action="store_true", help="decide right away (testing)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    Swing(dry=a.dry_run).run(now_mode=a.now)


if __name__ == "__main__":
    main()
