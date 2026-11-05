"""Main loop: gather data -> AI decides -> risk check -> place orders -> log.

Usage:
    python agent.py --once        # run a single cycle now
    python agent.py               # run forever, every INTERVAL_MINUTES while market is open
    python agent.py --dry-run     # full cycle but never send orders
    python agent.py --report      # print recent decisions from the journal
"""
import argparse
import logging
import time
from datetime import datetime, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, OrderSide, TimeInForce
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

import risk
from brain import Brain
from config import settings
from journal import Journal
from market_data import MarketData

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler("agent.log")])
log = logging.getLogger("agent")


class Agent:
    def __init__(self, dry_run=False):
        settings.validate()
        self.dry_run = dry_run
        self.trading = TradingClient(settings.alpaca_key, settings.alpaca_secret, paper=settings.paper)
        self.data = MarketData(settings.alpaca_key, settings.alpaca_secret, crypto=settings.crypto)
        self.brain = Brain(settings.ai_provider, settings.ai_key, settings.ai_model)
        self.journal = Journal(settings.db_path)
        log.info("Mode: %s%s | %s | AI: %s %s", "PAPER" if settings.paper else "*** LIVE ***",
                 " (dry run)" if dry_run else "", "CRYPTO" if settings.crypto else "STOCKS",
                 settings.ai_provider, settings.ai_model)

    def account(self):
        a = self.trading.get_account()
        return {"equity": float(a.equity), "last_equity": float(a.last_equity),
                "cash": float(a.cash), "buying_power": float(a.buying_power)}

    def positions(self):
        # Only show the AI positions of the asset class this run trades.
        # Alpaca reports crypto as "BTCUSD"; we use "BTC/USD" everywhere else.
        want = AssetClass.CRYPTO if settings.crypto else AssetClass.US_EQUITY
        def name(p):
            s = p.symbol
            return s if "/" in s or not settings.crypto else s[:-3] + "/USD"
        return {name(p): {"qty": float(p.qty), "market_value": float(p.market_value),
                           "avg_entry": float(p.avg_entry_price),
                           "unrealized_pl_%": round(float(p.unrealized_plpc) * 100, 2)}
                for p in self.trading.get_all_positions() if p.asset_class == want}

    def orders_today(self):
        """Count today's orders from Alpaca itself (works even without a local journal)."""
        start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        orders = self.trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, after=start, limit=100))
        mine = AssetClass.CRYPTO if settings.crypto else AssetClass.US_EQUITY
        return len([o for o in orders if o.asset_class == mine and str(o.status) not in ("OrderStatus.REJECTED", "rejected")])

    def recent_orders(self, n=20):
        orders = self.trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=n))
        return [{"time": str(o.submitted_at)[:16], "symbol": o.symbol, "side": str(o.side).split(".")[-1],
                 "qty": o.filled_qty, "avg_price": o.filled_avg_price,
                 "status": str(o.status).split(".")[-1]} for o in orders]

    def execute(self, a: risk.Approved):
        tif = TimeInForce.GTC if settings.crypto else TimeInForce.DAY  # crypto doesn't accept DAY
        if a.side == "buy":
            req = MarketOrderRequest(symbol=a.symbol, notional=a.notional,
                                     side=OrderSide.BUY, time_in_force=tif)
        else:
            req = MarketOrderRequest(symbol=a.symbol, qty=a.qty,
                                     side=OrderSide.SELL, time_in_force=tif)
        if self.dry_run:
            log.info("  [dry-run] would %s %s %s", a.side, a.symbol, a.notional or a.qty)
            return "dry-run", "dry-run"
        o = self.trading.submit_order(req)
        return str(o.id), str(o.status)

    def cycle(self):
        clock = self.trading.get_clock()
        market_open = True if settings.crypto else clock.is_open
        if not market_open:
            log.info("Market closed - nothing to do.")
            return
        acct, pos = self.account(), self.positions()
        symbols = sorted(set(settings.watchlist) | set(pos))

        context = {
            "now_utc": datetime.now(timezone.utc).isoformat(timespec="minutes"),
            "asset_class": "crypto (24/7, very volatile)" if settings.crypto else "US stocks/ETFs",
            "market_open": market_open,
            "account": acct,
            "day_pnl_%": round((acct["equity"] / acct["last_equity"] - 1) * 100, 2),
            "positions": pos,
            "watchlist_snapshots": self.data.snapshots(symbols),
            "news_last_48h": self.data.headlines(symbols),
            "your_recent_decisions": self.journal.recent(),
            "your_recent_orders": self.recent_orders(),
            "limits": {
                "max_position_%_of_equity": settings.max_position_pct * 100,
                "max_order_usd": settings.max_order_usd,
                "min_cash_reserve_%": settings.min_cash_reserve_pct * 100,
                "max_positions": settings.max_positions,
            },
        }
        try:
            result = self.brain.decide(context)
        except Exception as e:
            # AI temporarily unavailable (e.g. Google 503 "high demand"): skip this
            # cycle safely - no trades - and try again on the next scheduled run.
            log.warning("AI unavailable, skipping this cycle (no trades): %s", str(e)[:200])
            return
        cid = self.journal.cycle(acct, result)
        log.info("Market view: %s", result.get("market_view"))

        decisions = result.get("decisions", [])
        if not decisions:
            log.info("Decision: hold everything.")
            return

        approved, rejected = risk.check(decisions, acct, pos, max(self.journal.trades_today(), self.orders_today()),
                                        settings, market_open=market_open)
        for d, why in rejected:
            log.info("  REJECTED %s %s: %s", d.get("action"), d.get("symbol"), why)
            self.journal.decision(cid, d, "rejected", why)

        by_sym = {d["symbol"].upper(): d for d in decisions}
        for a in approved:
            try:
                oid, status = self.execute(a)
                log.info("  ORDER %s %s %s -> %s", a.side.upper(), a.symbol,
                         f"${a.notional}" if a.side == "buy" else f"{a.qty} sh", status)
                self.journal.order(a, oid, status)
                self.journal.decision(cid, by_sym[a.symbol], "executed",
                                      f"notional={a.notional} qty={a.qty}")
            except Exception as e:  # one failed order shouldn't kill the loop
                log.error("  ORDER FAILED %s %s: %s", a.side, a.symbol, e)
                self.journal.order(a, "", "error")
                self.journal.decision(cid, by_sym[a.symbol], "error", str(e))

    def run_forever(self):
        while True:
            try:
                clock = self.trading.get_clock()
                if settings.crypto or clock.is_open:
                    self.cycle()
                    time.sleep(settings.interval_minutes * 60)
                else:
                    wait = (clock.next_open - clock.timestamp).total_seconds()
                    log.info("Market closed. Next open %s (sleeping %.1fh).",
                             clock.next_open, wait / 3600)
                    time.sleep(min(max(wait, 60), 3600))
            except KeyboardInterrupt:
                log.info("Stopped by user."); break
            except Exception as e:
                log.exception("Cycle error: %s (retrying in 5 min)", e)
                time.sleep(300)


def report():
    j = Journal(settings.db_path)
    for r in reversed(j.recent(30)):
        print(f"{r['ts']}  {r['status']:<9} {r['action']:<4} {r['symbol']:<6} "
              f"conf={r['confidence']}  {r['reasoning'][:90]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--report", action="store_true")
    args = p.parse_args()
    if args.report:
        report()
    else:
        agent = Agent(dry_run=args.dry_run)
        agent.cycle() if args.once else agent.run_forever()
