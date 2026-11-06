"""Stop-losses and take-profits, enforced in code every cycle (the AI can't skip them).

When the AI buys, it gives a stop_loss_price and take_profit_price. They are saved in
memory/exits_<mode>.json (committed to the repo, so they survive between runs).
Before the AI is even asked, every open position is checked:
  - price <= stop          -> sell everything (cut the loser)
  - price >= take-profit   -> sell half, then move the stop up to the entry price
                              (the rest of the trade can no longer turn into a loss)
  - no stop on file        -> a default stop of DEFAULT_STOP_PCT below the entry price
Pure functions here (no network) so they can be tested.
"""
import json
import os

DIR = os.getenv("MEMORY_DIR", "memory")


def path(crypto):
    return os.path.join(DIR, f"exits_{'crypto' if crypto else 'stocks'}.json")


def load(crypto):
    try:
        with open(path(crypto)) as f:
            return json.load(f)
    except Exception:
        return {}


def save(crypto, levels):
    os.makedirs(DIR, exist_ok=True)
    with open(path(crypto), "w") as f:
        json.dump(levels, f, indent=1, sort_keys=True)


def record_buy(levels, symbol, entry, stop=None, take_profit=None, max_stop_pct=0.05):
    """Save exit levels for a new/added position. Silly levels are ignored:
    a stop must be below the entry and no more than max_stop_pct away;
    a take-profit must be above the entry."""
    cur = dict(levels.get(symbol, {}))
    try:
        stop = float(stop) if stop else None
        take_profit = float(take_profit) if take_profit else None
    except (TypeError, ValueError):
        stop = take_profit = None
    if stop and entry * (1 - max_stop_pct) <= stop < entry:
        cur["stop"] = round(stop, 4)
    if take_profit and take_profit > entry:
        cur["take_profit"] = round(take_profit, 4)
        cur.pop("took_profit", None)
    levels[symbol] = cur
    return levels


def check(positions, levels, default_stop_pct=0.03):
    """positions: {symbol: {"qty", "avg_entry", "price"}}
    Returns (actions, levels) where actions = [(symbol, sell_fraction, reason)]
    and levels has been updated (closed symbols removed, stops moved up)."""
    levels = {s: dict(v) for s, v in levels.items() if s in positions}  # forget closed positions
    actions = []
    for sym, p in positions.items():
        price, entry = float(p.get("price") or 0), float(p.get("avg_entry") or 0)
        if price <= 0 or entry <= 0:
            continue
        lv = levels.setdefault(sym, {})
        stop = lv.get("stop") or round(entry * (1 - default_stop_pct), 4)
        lv["stop"] = stop
        tp = lv.get("take_profit")
        if price <= stop:
            actions.append((sym, 1.0, f"STOP-LOSS hit: price {price:.2f} <= stop {stop:.2f}"))
            levels.pop(sym, None)
        elif tp and price >= tp and not lv.get("took_profit"):
            actions.append((sym, 0.5, f"TAKE-PROFIT hit: price {price:.2f} >= target {tp:.2f}, selling half"))
            lv["took_profit"] = True
            lv["stop"] = max(stop, round(entry, 4))  # rest of the trade can't become a loss
    return actions, levels
