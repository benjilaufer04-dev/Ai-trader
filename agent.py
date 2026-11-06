"""Main loop: gather data -> AI decides -> risk check -> place orders -> log.

Usage:
    python agent.py --once        # run a single cycle now
    python agent.py               # run forever, every INTERVAL_MINUTES while market is open
    python agent.py --dry-run     # full cycle but never send orders
    python agent.py --report      # print recent decisions from the journal
"""
import argparse
import json
import logging
import time
from datetime import datetime, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, OrderSide, TimeInForce
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

import exits
import learning
import risk
from brain import Brain
from config import settings
from journal import Journal
from market_data import MarketData

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler("agent.log")])
log = logging.getLogger("agent")


def owner_instructions():
    out = []
    if settings.trading_style == "active":
        out.append(
            "ACTIVE STYLE: the owner wants you engaged, but only with a real edge. "
            "Holding is fine when there is no clear setup - forced trades lose money. "
            "Prefer entries WITH the trend (price above a rising 20-day average) over "
            "catching falling knives. Open a small starter position (about 2-4% of equity), "
            "add only when it is working, never add to a loser. Every buy MUST include a "
            "stop_loss_price (1.5-4% below the current price) and a take_profit_price at least "
            "2x as far above the price as the stop is below it - code enforces them for you.")
    if settings.aggressive:
        out.append(
            "HIGH-RISK EXPERIMENT (paper money, owner's explicit choice): trade big and use leverage. "
            "The account has margin (buying_power, up to ~4x equity intraday) and the watchlist holds "
            "leveraged ETFs (3x/2x) and very volatile stocks. Size real conviction trades at 25-100% of "
            "equity each. Inverse ETFs (e.g. SQQQ, SOXS) are how you bet on a fall, since shorting is off. "
            "Still give every buy a stop_loss_price and take_profit_price - wider is fine (up to ~15%), "
            "and cut losers fast: leverage makes small mistakes big. Before the close, code cuts total "
            "exposure to under 2x equity, so prefer to be lighter going into the night.")
    if settings.focus:
        out.append("FOCUS: " + settings.focus)
    return out or ["Careful style: only act on clear setups."]


class Agent:
    def __init__(self, dry_run=False):
        settings.validate()
        self.dry_run = dry_run
        self.trading = TradingClient(settings.alpaca_key, settings.alpaca_secret, paper=settings.paper)
        self.data = MarketData(settings.alpaca_key, settings.alpaca_secret, crypto=settings.crypto)
        self.brain = Brain(settings.ai_provider, settings.ai_key, settings.ai_model)
        self.journal = Journal(settings.db_path)
        log.info("Mode: %s%s%s | %s | AI: %s %s", "PAPER" if settings.paper else "*** LIVE ***",
                 " | HIGH-RISK" if settings.aggressive else "",
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
                           "price": float(p.current_price),
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

    @staticmethod
    def part_qty(held, frac):
        """Quantity for a partial sell (fractional; execute() falls back to whole shares if needed)."""
        return int(held * frac * 1e6) / 1e6

    def execute(self, a: risk.Approved):
        tif = TimeInForce.GTC if settings.crypto else TimeInForce.DAY  # crypto doesn't accept DAY
        if a.side == "buy":
            req = MarketOrderRequest(symbol=a.symbol, notional=a.notional,
                                     side=OrderSide.BUY, time_in_force=tif)
        else:
            req = MarketOrderRequest(symbol=a.symbol, qty=a.qty,
                                     side=OrderSide.SELL, time_in_force=tif)
        if a.side == "sell" and self.dry_run is False:
            held = self.positions().get(a.symbol, {}).get("qty", 0)
            if held and a.qty >= held:  # full exit: let Alpaca close the exact position
                o = self.trading.close_position(a.symbol.replace("/", ""))
                return str(o.id), str(o.status)
        if self.dry_run:
            log.info("  [dry-run] would %s %s %s", a.side, a.symbol, a.notional or a.qty)
            return "dry-run", "dry-run"
        try:
            o = self.trading.submit_order(req)
        except Exception as e:
            # Some symbols (several leveraged ETFs) can't be bought in fractions -> buy whole shares.
            if "fraction" not in str(e).lower():
                raise
            if a.side == "sell":  # non-fractionable position: sell whole shares
                shares = max(1, int(round(a.qty)))
                o = self.trading.submit_order(MarketOrderRequest(symbol=a.symbol, qty=shares,
                                                                 side=OrderSide.SELL, time_in_force=tif))
                return str(o.id), str(o.status)
            price = (self.data.snapshots([a.symbol]).get(a.symbol) or {}).get("price") or 0
            shares = int(a.notional // price) if price else 0
            if shares < 1:
                raise
            log.info("  %s not fractionable -> buying %d whole shares", a.symbol, shares)
            o = self.trading.submit_order(MarketOrderRequest(symbol=a.symbol, qty=shares,
                                                             side=OrderSide.BUY, time_in_force=tif))
        return str(o.id), str(o.status)

    def cycle(self):
        clock = self.trading.get_clock()
        market_open = True if settings.crypto else clock.is_open
        if not market_open:
            log.info("Market closed - nothing to do.")
            return
        self._enforce_exits()
        if settings.aggressive:
            self._delever_before_close(clock)
        acct, pos = self.account(), self.positions()
        symbols = sorted(set(settings.watchlist) | set(pos))
        snapshots = self.data.snapshots(symbols)
        prices = {s: v["price"] for s, v in snapshots.items()}
        past = learning.load_decisions(settings.crypto, days=14)

        context = {
            "now_utc": datetime.now(timezone.utc).isoformat(timespec="minutes"),
            "asset_class": "crypto (24/7, very volatile)" if settings.crypto else "US stocks/ETFs",
            "market_open": market_open,
            "account": acct,
            "day_pnl_%": round((acct["equity"] / acct["last_equity"] - 1) * 100, 2),
            "positions": pos,
            "watchlist_snapshots": snapshots,
            "news_last_48h": self.data.headlines(symbols),
            "owner_instructions": owner_instructions(),
            "lessons_learned": learning.load_lessons(),
            "track_record": learning.track_record(learning.outcomes(past, prices)),
            "your_recent_decisions": self.journal.recent(),
            "your_recent_orders": self.recent_orders(),
            "exit_levels_enforced_by_code": exits.load(settings.crypto),
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
        statuses = {}
        try:
            self._trade(cid, decisions, acct, pos, market_open, statuses)
        finally:
            learning.log_decisions(settings.crypto, result.get("market_view", ""), decisions, statuses, snapshots)

    def _enforce_exits(self):
        """Sell positions that hit their stop-loss or take-profit. Runs before the AI,
        ignores the daily trade limit (cutting a loser must always be possible)."""
        pos = self.positions()
        actions, levels = exits.check(pos, exits.load(settings.crypto), settings.default_stop_pct)
        for sym, frac, why in actions:
            held = pos[sym]["qty"]
            qty = held if frac >= 0.999 else self.part_qty(held, frac)
            a = risk.Approved(sym, "sell", qty=qty, reasoning=why)
            try:
                oid, status = self.execute(a)
                log.info("  EXIT %s %s -> %s (%s)", sym, qty, status, why)
                self.journal.order(a, oid, status)
                learning.log_decisions(settings.crypto, "auto exit (code)", [
                    {"symbol": sym, "action": "sell", "sell_fraction": frac, "confidence": 1,
                     "reasoning": why}], {sym: "executed"}, {sym: {"price": pos[sym]["price"]}})
            except Exception as e:
                log.error("  EXIT FAILED %s: %s", sym, e)
        exits.save(settings.crypto, levels)

    def _delever_before_close(self, clock):
        """Leverage: Alpaca allows ~4x during the day but only 2x overnight.
        Near the close, trim every position by the same fraction to get under OVERNIGHT_LEVERAGE."""
        if settings.crypto or not clock.next_close:
            return
        minutes_left = (clock.next_close - clock.timestamp).total_seconds() / 60
        if minutes_left > settings.delever_minutes_before_close:
            return
        acct, pos = self.account(), self.positions()
        gross = sum(abs(p["market_value"]) for p in pos.values())
        target = acct["equity"] * settings.overnight_leverage
        if gross <= target or gross <= 0:
            return
        frac = min(1.0, (gross - target) / gross + 0.02)
        log.info("DELEVER before close: exposure %.0f > %.0f (%.1fx), trimming %.0f%% of each position",
                 gross, target, gross / acct["equity"], frac * 100)
        for sym, p in pos.items():
            qty = self.part_qty(p["qty"], frac)
            if qty <= 0:
                continue
            try:
                self.execute(risk.Approved(sym, "sell", qty=qty, reasoning="overnight leverage limit"))
                log.info("  TRIM %s %s", sym, qty)
            except Exception as e:
                log.error("  TRIM FAILED %s: %s", sym, e)

    def _trade(self, cid, decisions, acct, pos, market_open, statuses):
        if not decisions:
            log.info("Decision: hold everything.")
            return

        approved, rejected = risk.check(decisions, acct, pos, max(self.journal.trades_today(), self.orders_today()),
                                        settings, market_open=market_open)
        for d, why in rejected:
            log.info("  REJECTED %s %s: %s", d.get("action"), d.get("symbol"), why)
            self.journal.decision(cid, d, "rejected", why)
            statuses[str(d.get("symbol", "")).upper()] = "rejected: " + why

        by_sym = {d["symbol"].upper(): d for d in decisions}
        for a in approved:
            try:
                oid, status = self.execute(a)
                log.info("  ORDER %s %s %s -> %s", a.side.upper(), a.symbol,
                         f"${a.notional}" if a.side == "buy" else f"{a.qty} sh", status)
                self.journal.order(a, oid, status)
                self.journal.decision(cid, by_sym[a.symbol], "executed",
                                      f"notional={a.notional} qty={a.qty}")
                statuses[a.symbol] = "executed"
                if a.side == "buy" and not self.dry_run:
                    d = by_sym[a.symbol]
                    price = (self.data.snapshots([a.symbol]).get(a.symbol) or {}).get("price") or 0
                    if price:
                        lv = exits.record_buy(exits.load(settings.crypto), a.symbol, float(price),
                                              d.get("stop_loss_price"), d.get("take_profit_price"),
                                              settings.max_stop_pct)
                        exits.save(settings.crypto, lv)
                        log.info("  EXITS %s: %s", a.symbol, lv.get(a.symbol))
            except Exception as e:  # one failed order shouldn't kill the loop
                log.error("  ORDER FAILED %s %s: %s", a.side, a.symbol, e)
                self.journal.order(a, "", "error")
                self.journal.decision(cid, by_sym[a.symbol], "error", str(e))
                statuses[a.symbol] = "error"

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


def status():
    """Print account value, today's P/L and every open position (no trading)."""
    settings.validate()
    t = TradingClient(settings.alpaca_key, settings.alpaca_secret, paper=settings.paper)
    a = t.get_account()
    eq, last = float(a.equity), float(a.last_equity)
    gross = sum(abs(float(p.market_value)) for p in t.get_all_positions())
    print(f"LEVERAGE {gross/eq:.2f}x (positions ${gross:,.0f} on equity ${eq:,.0f}) | buying power ${float(a.buying_power):,.0f}")
    print(f"MARGIN   multiplier {a.multiplier} | day-trading buying power ${float(a.daytrading_buying_power or 0):,.0f} | "
          f"overnight (RegT) buying power ${float(a.regt_buying_power or 0):,.0f} | day trades (5d) {a.daytrade_count}")
    print(f"ACCOUNT  value ${eq:,.2f} | today {eq-last:+,.2f} ({(eq/last-1)*100:+.2f}%) | "
          f"total since start {eq-100000:+,.2f} ({(eq/100000-1)*100:+.2f}%) | cash ${float(a.cash):,.2f}")
    pos = t.get_all_positions()
    if not pos:
        print("POSITIONS none")
    for p in pos:
        print(f"POSITION {p.symbol:<8} qty {float(p.qty):g} | bought @ ${float(p.avg_entry_price):,.2f} | "
              f"now ${float(p.current_price):,.2f} | value ${float(p.market_value):,.2f} | "
              f"P/L {float(p.unrealized_pl):+,.2f} ({float(p.unrealized_plpc)*100:+.2f}%)")
    for o in t.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=10)):
        print(f"ORDER    {str(o.submitted_at)[:16]} {str(o.side).split('.')[-1]} {o.symbol} "
              f"{str(o.status).split('.')[-1]} filled {o.filled_qty} @ {o.filled_avg_price}")


def cleanup(keep):
    """Close every position whose symbol is not in `keep` (stocks queue for the next open)."""
    settings.validate()
    t = TradingClient(settings.alpaca_key, settings.alpaca_secret, paper=settings.paper)
    keep = {s.strip().upper() for s in keep.split(",") if s.strip()}
    for p in t.get_all_positions():
        if p.symbol.upper() in keep:
            print(f"KEEP     {p.symbol}")
            continue
        try:
            o = t.close_position(p.symbol)
            print(f"CLOSE    {p.symbol} qty {float(p.qty):g} (P/L {float(p.unrealized_pl):+,.2f}) -> {str(o.status).split('.')[-1]}")
        except Exception as e:
            print(f"FAILED   {p.symbol}: {str(e)[:150]}")


def daily_review():
    """After the close: score recent decisions, ask the AI what it learned, save lessons."""
    settings.validate()
    from market_data import MarketData
    from brain import Brain
    t = TradingClient(settings.alpaca_key, settings.alpaca_secret, paper=settings.paper)
    acc = t.get_account()
    eq, last = float(acc.equity), float(acc.last_equity)
    account_line = (f"Account ${eq:,.2f} | today {eq-last:+,.2f} ({(eq/last-1)*100:+.2f}%) | "
                    f"since start {eq-100000:+,.2f} ({(eq/100000-1)*100:+.2f}%)")
    positions = [{"symbol": p.symbol, "value": round(float(p.market_value), 2),
                  "pl_%": round(float(p.unrealized_plpc) * 100, 2)} for p in t.get_all_positions()]

    decisions = learning.load_decisions(None, days=7)
    stock_syms = sorted({d["symbol"] for d in decisions if d.get("mode") == "stocks" and d.get("symbol")} | {"GLD", "SPY"})
    crypto_syms = sorted({d["symbol"] for d in decisions if d.get("mode") == "crypto" and d.get("symbol")})
    prices = {}
    for crypto, syms in ((False, stock_syms), (True, crypto_syms)):
        if syms:
            try:
                snaps = MarketData(settings.alpaca_key, settings.alpaca_secret, crypto=crypto).snapshots(syms)
                prices.update({s: v["price"] for s, v in snaps.items()})
            except Exception as e:
                log.warning("price fetch failed (%s): %s", "crypto" if crypto else "stocks", e)
    rows = learning.outcomes(decisions, prices)
    holds = sum(1 for d in decisions if d.get("action") == "hold_all")
    rejected = [{"ts": d["ts"][:16], "symbol": d.get("symbol"), "action": d.get("action"),
                 "why": d.get("status")} for d in decisions if str(d.get("status", "")).startswith(("rejected", "error"))]
    state = {
        "account": account_line, "positions_now": positions,
        "benchmark": {"GLD_now": prices.get("GLD"), "SPY_now": prices.get("SPY"),
                      "note": "GLD = just holding gold; SPY = just holding the S&P 500"},
        "scored_decisions": rows, "track_record": learning.track_record(rows),
        "cycles_where_it_held_everything": holds,
        "rejected_or_failed_orders": rejected[-20:],
        "current_lessons": learning.load_lessons(),
        "owner_instructions": owner_instructions(),
    }
    brain = Brain(settings.ai_provider, settings.ai_key, settings.ai_model)
    review = brain.ask_json(learning.REVIEW_SYSTEM, "Review data (JSON):\n" + json.dumps(state, indent=1, default=str),
                            learning.REVIEW_TOOL)
    lessons = learning.save_review(review, account_line)
    print(account_line)
    print("SUMMARY:", review.get("summary", ""))
    for m in review.get("mistakes", []):
        print("MISTAKE:", m)
    for l in lessons:
        print("LESSON:", l)


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
    p.add_argument("--status", action="store_true")
    p.add_argument("--review", action="store_true")
    p.add_argument("--cleanup", metavar="KEEP", help="close all positions except these, e.g. GLD,SPY,QQQ")
    args = p.parse_args()
    if args.status:
        status()
    elif args.cleanup:
        cleanup(args.cleanup)
    elif args.review:
        daily_review()
    elif args.report:
        report()
    else:
        agent = Agent(dry_run=args.dry_run)
        agent.cycle() if args.once else agent.run_forever()
