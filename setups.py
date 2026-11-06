"""Expert-style day-trading setups, backtested with the same honesty rules as backtest.py.

Both setups only trade WITH the whole market (QQQ) - the main thing pros check first:
  market UP   (QQQ above its VWAP and above today's open) -> only long funds: TQQQ, SOXL, TSLA, NVDA
  market DOWN (QQQ below its VWAP and below today's open) -> only inverse funds: SQQQ, SOXS

ORB  - Opening Range Breakout. The high/low of the first N minutes is the "opening range".
       Buy when a 1-min bar closes above the range high on above-average volume.
       Stop: range low (or range middle). Target: k x risk. One trade per symbol per day.
VWAP - Trend pullback. On a trend day, after the stock has stayed above VWAP, wait for a dip
       back to VWAP that holds, then buy the first green bar that takes out the previous bar's high.
       Stop: just under VWAP / the dip low. Target: k x risk. Max 2 per symbol per day.

Sizing like a pro: risk a fixed % of the account per trade (money lost if the stop hits),
position = risk / (entry - stop), capped at MAX_X x equity (leverage cap).
Costs: 0.03% slippage per side, stops gap (a bar opening below the stop fills at the open),
stop assumed first if a bar touches both. Flat before the close.

Usage: python setups.py --days 60
"""
import argparse
import itertools
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import backtest as bt

LONGS, INVERSES = ["TQQQ", "SOXL", "TSLA", "NVDA"], ["SQQQ", "SOXS"]
SLIP = 0.0003


def day_arrays(df):
    df = df.reset_index()
    c, v = df["close"], df["volume"]
    tp = (df["high"] + df["low"] + c) / 3
    vwap = (tp * v).cumsum() / v.cumsum().replace(0, np.nan)
    ny = df["timestamp"].dt.tz_convert("America/New_York")
    return {"o": df["open"].values, "h": df["high"].values, "l": df["low"].values, "c": c.values,
            "v": v.values, "vwap": vwap.values, "minute": (ny.dt.hour * 60 + ny.dt.minute - 570).values,
            "vavg": v.rolling(20).mean().shift(1).values}


def at(a, m):
    """index of minute m in arrays a, or None"""
    i = np.searchsorted(a["minute"], m)
    return i if i < len(a["minute"]) and a["minute"][i] == m else None


def regime(q, m):
    """+1 market up, -1 down, 0 unclear, using QQQ bar that closed at minute m"""
    i = at(q, m)
    if i is None:
        return 0
    c = q["c"][i]
    if c > q["vwap"][i] and c > q["o"][0]:
        return 1
    if c < q["vwap"][i] and c < q["o"][0]:
        return -1
    return 0


def orb_entries(a, q, sym, cfg):
    n = cfg["or_min"]
    first = a["minute"] < n
    if first.sum() < n * 0.8:
        return []
    hi, lo = a["h"][first].max(), a["l"][first].min()
    mid = (hi + lo) / 2
    for i in range(len(a["c"]) - 1):
        m = a["minute"][i]
        if m < n or m > cfg["last_entry_min"]:
            continue
        if a["c"][i] > hi and a["c"][i] > a["o"][i] and a["v"][i] >= cfg["vol_mult"] * (a["vavg"][i] or np.inf):
            want = 1 if sym in LONGS else -1
            if cfg["filter"] and regime(q, m) != want:
                return []          # first breakout came against the market -> skip the day
            stop = lo if cfg["orb_stop"] == "low" else mid
            return [(m + 1, i + 1, stop)]
        if a["c"][i] < lo:
            return []              # broke down first: no long setup today
    return []


def vwap_entries(a, q, sym, cfg):
    out, above_streak, dipped, dip_low = [], 0, False, None
    want = 1 if sym in LONGS else -1
    for i in range(1, len(a["c"]) - 1):
        m, c, vw = a["minute"][i], a["c"][i], a["vwap"][i]
        if np.isnan(vw):
            continue
        if not dipped:
            above_streak = above_streak + 1 if c > vw else 0
            if above_streak >= cfg["trend_bars"] and a["l"][i] <= vw * (1 + cfg["touch"]) and m >= 30:
                dipped, dip_low = True, a["l"][i]
            continue
        dip_low = min(dip_low, a["l"][i])
        if c < vw * (1 - cfg["touch"]):          # lost VWAP: setup failed, reset
            dipped, above_streak = False, 0
            continue
        if c > a["o"][i] and c > a["h"][i - 1] and m <= cfg["last_entry_min"]:
            if not cfg["filter"] or regime(q, m) == want:
                stop = min(dip_low, vw * (1 - cfg["touch"])) * 0.999
                if stop < c:
                    out.append((m + 1, i + 1, stop))
            dipped, above_streak = False, 0
            if len(out) >= 2:
                break
    return out


def simulate(bars, days, cfg, equity=100_000):
    trades = []
    for day in days:
        arr = {}
        for s, x in bars.items():
            d = x[x["day"] == day]
            if len(d) > 60:
                arr[s] = day_arrays(d.drop(columns="day"))
        if "QQQ" not in arr:
            continue
        q = arr["QQQ"]
        cands = []
        for s in LONGS + INVERSES:
            if s not in arr:
                continue
            f = orb_entries if cfg["setup"] == "orb" else vwap_entries
            for m, i, stop in f(arr[s], q, s, cfg):
                if i < len(arr[s]["o"]):
                    cands.append((m, s, i, stop))
        cands.sort()
        busy_until = []   # list of (exit_minute) for open positions -> max concurrent
        for m, s, i, stop in cands:
            busy_until = [e for e in busy_until if e > m]
            if len(busy_until) >= cfg["max_positions"]:
                continue
            a = arr[s]
            entry = a["o"][i] * (1 + SLIP)
            risk = entry - stop
            if risk <= entry * 0.001:
                continue
            usd = min(equity * cfg["risk_pct"] / (risk / entry), equity * cfg["max_x"])
            target = entry + cfg["rr"] * risk
            cur_stop, exit_px, why, j = stop, None, None, i
            for j in range(i, len(a["c"])):
                o, h, l, c = a["o"][j], a["h"][j], a["l"][j], a["c"][j]
                if l <= cur_stop:
                    exit_px, why = min(o, cur_stop), "stop"; break
                if h >= target:
                    exit_px, why = max(o, target), "target"; break
                if cfg["breakeven_r"] and h >= entry + cfg["breakeven_r"] * risk:
                    cur_stop = max(cur_stop, entry)
                if a["minute"][j] >= 380:
                    exit_px, why = c, "eod"; break
            if exit_px is None:
                exit_px, why = a["c"][-1], "eod"
            exit_px *= (1 - SLIP)
            trades.append({"day": str(day), "symbol": s, "entry": entry, "exit": exit_px, "usd": usd,
                           "pnl": (exit_px / entry - 1) * usd, "why": why, "minutes": int(a["minute"][j] - m)})
            busy_until.append(a["minute"][j])
    return trades


def grid():
    base = {"max_positions": 2, "risk_pct": 0.005, "max_x": 2.0, "last_entry_min": 180}
    for n, st, rr, vm, f, be in itertools.product([5, 15, 30], ["low", "mid"], [1.5, 2.0, 3.0],
                                                  [1.0, 1.5, 2.5], [1, 0], [0, 1.0]):
        yield dict(base, setup="orb", or_min=n, orb_stop=st, rr=rr, vol_mult=vm, filter=f, breakeven_r=be)
    for tb, touch, rr, f, be in itertools.product([15, 30, 60], [0.001, 0.003], [1.5, 2.0, 3.0], [1, 0], [0, 1.0]):
        yield dict(base, setup="vwap", trend_bars=tb, touch=touch, rr=rr, filter=f, breakeven_r=be)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    a = ap.parse_args()
    bars, days = bt.load_bars(LONGS + INVERSES + ["QQQ"], a.days)
    split = int(len(days) * 2 / 3)
    train, test = days[:split], days[split:]
    rows = []
    for cfg in grid():
        s = bt.stats(simulate(bars, train, cfg), train)
        score = s["pnl"] + 0.5 * s["max_drawdown"] if s["trades"] >= 20 else -1e9
        rows.append((score, s, cfg))
    rows.sort(key=lambda r: r[0], reverse=True)
    lines = [f"# Expert setups backtest {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
             f"{len(days)} days {days[0]}..{days[-1]} | train {train[0]}..{train[-1]} | test {test[0]}..{test[-1]}",
             "Risk 0.5% of the account per trade, max 2x leverage, max 2 trades at once, 0.03% slippage/side.", ""]
    best = {}
    for setup in ("orb", "vwap"):
        lines.append(f"## {setup.upper()}: best 5 on TRAIN, then TEST (unseen)")
        for _, s_train, cfg in [r for r in rows if r[2]["setup"] == setup][:5]:
            s_test = bt.stats(simulate(bars, test, cfg), test)
            keys = {k: v for k, v in cfg.items() if k not in ("max_positions", "risk_pct", "max_x", "last_entry_min")}
            lines += [json.dumps(keys), f"- TRAIN: {s_train}", f"- TEST : {s_test}", ""]
            ok = s_train["pnl"] > 0 and s_test["pnl"] > 0 and s_test["trades"] >= 8
            if ok and setup not in best:
                best[setup] = {"params": cfg, "train": s_train, "test": s_test}
    lines += ["## Passed (profitable on BOTH train and unseen test)", json.dumps(best, default=str) if best else "none"]
    os.makedirs(bt.REPORT_DIR, exist_ok=True)
    open(os.path.join(bt.REPORT_DIR, f"setups_{datetime.now(timezone.utc):%Y-%m-%d_%H%M}.md"), "w").write("\n".join(lines) + "\n")
    json.dump(best, open(os.path.join(bt.REPORT_DIR, "setups_latest.json"), "w"), indent=1, default=str)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
