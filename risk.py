"""Hard guardrails between the AI and the broker. Pure functions, no network, fully testable.

The AI proposes; this module disposes. Every proposal is either approved
(possibly resized down) or rejected with a reason.
"""
import os
from dataclasses import dataclass


@dataclass
class Approved:
    symbol: str
    side: str            # "buy" | "sell"
    notional: float = 0  # buys
    qty: float = 0       # sells
    reasoning: str = ""


def check(decisions, account, positions, trades_today, settings, market_open=True):
    """
    account:   {"equity", "last_equity", "cash", "buying_power"}
    positions: {symbol: {"qty", "market_value"}}
    trades_today: count of orders already placed today
    Returns (approved: list[Approved], rejected: list[(decision, reason)])
    """
    approved, rejected = [], []

    def reject(d, why):
        rejected.append((d, why))

    if os.path.exists(settings.kill_switch_file):
        return [], [(d, "kill switch file present") for d in decisions]
    if not market_open:
        return [], [(d, "market closed") for d in decisions]

    equity = float(account["equity"])
    day_pnl_pct = equity / float(account["last_equity"]) - 1
    buys_blocked = day_pnl_pct <= -settings.daily_loss_limit_pct
    cash = float(account["cash"])
    min_cash = equity * settings.min_cash_reserve_pct
    n_positions = len(positions)
    trades_left = settings.max_trades_per_day - trades_today
    seen = set()

    # Process sells first so they free up cash/slots for buys.
    ordered = sorted(decisions, key=lambda d: d.get("action") != "sell")

    for d in ordered:
        sym = str(d.get("symbol", "")).upper()
        action = d.get("action")
        conf = float(d.get("confidence", 0))

        if trades_left <= 0:
            reject(d, "daily trade limit reached"); continue
        if sym not in settings.watchlist:
            reject(d, "symbol not on watchlist"); continue
        if sym in seen:
            reject(d, "duplicate symbol this cycle"); continue
        if conf < settings.min_confidence:
            reject(d, f"confidence {conf:.2f} < {settings.min_confidence}"); continue

        if action == "sell":
            pos = positions.get(sym)
            if not pos or float(pos["qty"]) <= 0:
                reject(d, "no position to sell (shorting not allowed)"); continue
            frac = min(max(float(d.get("sell_fraction", 1.0)), 0.0), 1.0)
            qty = round(float(pos["qty"]) * frac, 6)
            if qty <= 0:
                reject(d, "sell quantity is zero"); continue
            approved.append(Approved(sym, "sell", qty=qty, reasoning=d.get("reasoning", "")))
            cash += float(pos["market_value"]) * frac
            if frac >= 0.999:
                n_positions -= 1

        elif action == "buy":
            if buys_blocked:
                reject(d, f"daily loss limit hit ({day_pnl_pct:.2%})"); continue
            held = float(positions.get(sym, {}).get("market_value", 0))
            if sym not in positions and n_positions >= settings.max_positions:
                reject(d, "max number of positions reached"); continue
            want = float(d.get("buy_usd", 0))
            room_position = equity * settings.max_position_pct - held
            room_cash = cash - min_cash
            notional = round(min(want, settings.max_order_usd, room_position, room_cash), 2)
            if notional < 1:
                reject(d, "no room (position cap / cash reserve)"); continue
            approved.append(Approved(sym, "buy", notional=notional, reasoning=d.get("reasoning", "")))
            cash -= notional
            if sym not in positions:
                n_positions += 1
        else:
            reject(d, f"unknown action {action!r}"); continue

        seen.add(sym)
        trades_left -= 1

    return approved, rejected
