"""Backtest + self-tuning for the scalper.

Replays past 1-minute bars with the SAME rules as scalper.py (same indicator math),
including realistic costs:
  - entry at the NEXT bar's open after the signal bar (you can't buy the close you just saw)
  - slippage SLIP per side (default 0.03%)
  - stop orders that gap: if a bar opens below the stop you get the open, not the stop
  - if a bar touches both stop and target, assume the stop hit first (pessimistic)
  - max positions at once, cooldown after an exit, flat before the close

Walk-forward tuning (to avoid fooling ourselves):
  days are split into TRAIN (first 2/3) and TEST (last 1/3). Settings are chosen on TRAIN
  only, then judged on TEST, which they have never seen. New settings are adopted only if they
  make money on TEST and beat the current settings there.

Usage:
  python backtest.py --days 30                 # baseline report for current settings
  python backtest.py --days 30 --tune          # + search better settings
  python backtest.py --days 30 --tune --adopt  # + write memory/scalper_params.json if they pass
"""
import argparse
import itertools
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import scalper as sc

PARAMS_FILE = os.path.join(os.getenv("MEMORY_DIR", "memory"), "scalper_params.json")
REPORT_DIR = os.path.join(os.getenv("MEMORY_DIR", "memory"), "backtests")
# "Do the opposite": when the rules say buy X, buy the fund that moves the other way.
OPPOSITE = {"TQQQ": "SQQQ", "SQQQ": "TQQQ", "SOXL": "SOXS", "SOXS": "SOXL"}

TUNABLE = ["stop_pct", "target_pct", "vol_mult", "max_extension", "time_stop_min", "position_x",
           "momentum_exit", "entry_window_min"]


# ---------- data ----------

def load_bars(symbols, days):
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    c = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    end = datetime.now(timezone.utc) - timedelta(minutes=16)
    start = end - timedelta(days=int(days * 1.5) + 4)
    df = c.get_stock_bars(StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Minute,
                                           start=start, end=end, feed=DataFeed.IEX)).df
    out = {}
    for s in symbols:
        if s not in df.index.get_level_values(0):
            continue
        x = df.xs(s).sort_index()[["open", "high", "low", "close", "volume"]]
        ny = x.index.tz_convert("America/New_York")
        x = x[(ny.time >= pd.Timestamp("09:30").time()) & (ny.time < pd.Timestamp("16:00").time())]
        x["day"] = x.index.tz_convert("America/New_York").date
        out[s] = x
    all_days = sorted({d for x in out.values() for d in x["day"].unique()})[-days:]
    return {s: x[x["day"].isin(all_days)] for s, x in out.items()}, all_days


# ---------- per-day precomputation (same math as scalper.indicators / entry_signal) ----------

def prep_day(df, cfg):
    df = df.reset_index()
    ind = sc.indicators(df)
    c, o, v = df["close"], df["open"], df["volume"]
    vavg_prev = ind["vol_avg"].shift(1)
    prior_high = df["high"].shift(1).rolling(5).max()
    ext = c / ind["vwap"] - 1
    sig = ((c > ind["vwap"]) & (ind["ema9"] > ind["ema21"]) & (c > prior_high) & (c > o)
           & (vavg_prev > 0) & (v >= cfg["vol_mult"] * vavg_prev) & (ext <= cfg["max_extension"]))
    sig &= pd.Series(np.arange(len(df)) >= 24)
    score = (v / vavg_prev).where(sig, 0.0)
    dead = (ind["ema9"] < ind["ema21"]) & (c < ind["vwap"]) & pd.Series(np.arange(len(df)) >= 21)
    minute = (df["timestamp"].dt.tz_convert("America/New_York").dt.hour * 60
              + df["timestamp"].dt.tz_convert("America/New_York").dt.minute - 570)  # minutes since 9:30
    return {"o": o.values, "h": df["high"].values, "l": df["low"].values, "c": c.values,
            "score": score.fillna(0).values, "dead": dead.values, "minute": minute.values,
            "ts": df["timestamp"].values}


def simulate(bars, days, cfg, equity=100_000, slip=0.0003):
    """Event-driven over all symbols, minute by minute. Returns list of trades."""
    trades = []
    session_len = 390
    for day in days:
        prepped = {}
        for s, x in bars.items():
            d = x[x["day"] == day]
            if len(d) > 30:
                prepped[s] = prep_day(d.drop(columns="day"), cfg)
        if not prepped:
            continue
        # align by minute-of-session
        open_pos, cooldown = {}, {}
        for m in range(session_len):
            # 1) manage open positions on this bar
            for s in list(open_pos):
                p = prepped[s]
                idx = np.searchsorted(p["minute"], m)
                if idx >= len(p["minute"]) or p["minute"][idx] != m:
                    continue
                t = open_pos[s]
                o, h, l, c = p["o"][idx], p["h"][idx], p["l"][idx], p["c"][idx]
                exit_px, why = None, None
                if l <= t["stop"]:
                    exit_px, why = min(o, t["stop"]), "stop"
                elif h >= t["target"]:
                    exit_px, why = max(o, t["target"]), "target"
                else:
                    t["best"] = max(t["best"], h)
                    g = t["best"] / t["entry"] - 1
                    if g >= cfg["lock_at"]:
                        t["stop"] = max(t["stop"], t["entry"] * (1 + cfg["lock_pct"]))
                    elif g >= cfg["breakeven_at"]:
                        t["stop"] = max(t["stop"], t["entry"])
                    age = m - t["m"]
                    if age >= cfg["time_stop_min"] and g < cfg["time_stop_needs"]:
                        exit_px, why = c, "time"
                    elif cfg["momentum_exit"] and p["dead"][idx]:
                        exit_px, why = c, "momentum"
                    elif m >= session_len - cfg["flatten_before_close_min"]:
                        exit_px, why = c, "eod"
                if exit_px is not None:
                    exit_px *= (1 - slip)
                    trades.append({"day": str(day), "symbol": s, "entry": t["entry"], "exit": exit_px,
                                   "usd": t["usd"], "pnl": (exit_px / t["entry"] - 1) * t["usd"],
                                   "why": why, "minutes": m - t["m"]})
                    del open_pos[s]
                    cooldown[s] = m + cfg["cooldown_min"]
            # 2) entries: signal on the bar that closed at minute m-1 -> buy this bar's open
            if (m - 1 < cfg["skip_open_min"] or m > session_len - cfg["no_new_before_close_min"]
                    or m > cfg["entry_window_min"]
                    or len(open_pos) >= cfg["max_positions"]):
                continue
            cands = []
            for s, p in prepped.items():
                if s in open_pos or cooldown.get(s, -1) > m:
                    continue
                idx = np.searchsorted(p["minute"], m - 1)
                if idx >= len(p["minute"]) - 1 or p["minute"][idx] != m - 1 or p["score"][idx] <= 0:
                    continue
                if p["minute"][idx + 1] != m:
                    continue
                if cfg.get("inverse"):
                    s2 = OPPOSITE.get(s)          # no opposite fund (TSLA, NVDA) -> skip
                    if not s2 or s2 in open_pos or s2 not in prepped:
                        continue
                    p2 = prepped[s2]
                    j = np.searchsorted(p2["minute"], m)
                    if j >= len(p2["minute"]) or p2["minute"][j] != m:
                        continue
                    cands.append((p["score"][idx], s2, p2["o"][j]))
                    continue
                cands.append((p["score"][idx], s, p["o"][idx + 1]))
            for score, s, px in sorted(cands, reverse=True):
                if len(open_pos) >= cfg["max_positions"]:
                    break
                entry = px * (1 + slip)
                open_pos[s] = {"entry": entry, "stop": entry * (1 - cfg["stop_pct"]),
                               "target": entry * (1 + cfg["target_pct"]), "best": entry, "m": m,
                               "usd": equity * cfg["position_x"]}
    return trades


def stats(trades, days):
    if not trades:
        return {"trades": 0, "pnl": 0, "win_rate": 0, "profit_factor": 0, "max_drawdown": 0,
                "avg_win": 0, "avg_loss": 0, "per_day": 0, "green_days": "0/0"}
    df = pd.DataFrame(trades)
    wins, losses = df[df.pnl > 0], df[df.pnl <= 0]
    eq = df["pnl"].cumsum()
    by_day = df.groupby("day")["pnl"].sum()
    r = {
        "trades": len(df), "pnl": round(df.pnl.sum(), 0),
        "win_rate": round(len(wins) / len(df), 3),
        "profit_factor": round(wins.pnl.sum() / -losses.pnl.sum(), 2) if len(losses) and losses.pnl.sum() < 0 else 99,
        "max_drawdown": round(float((eq - eq.cummax()).min()), 0),
        "avg_win": round(wins.pnl.mean(), 0) if len(wins) else 0,
        "avg_loss": round(losses.pnl.mean(), 0) if len(losses) else 0,
        "per_day": round(df.pnl.sum() / max(1, len(days)), 0),
        "green_days": f"{int((by_day > 0).sum())}/{len(days)}",
        "exits": {k: int(v) for k, v in df["why"].value_counts().items()},
    }
    return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in r.items()}


def current_cfg():
    cfg = dict(sc.CFG)
    if os.path.exists(PARAMS_FILE):
        cfg.update(json.load(open(PARAMS_FILE)).get("params", {}))
    return cfg


def with_params(base, **kw):
    cfg = dict(base, **kw)
    # trailing rules scale with the stop
    cfg["breakeven_at"] = cfg["stop_pct"]
    cfg["lock_at"] = cfg["stop_pct"] * 1.5
    cfg["lock_pct"] = cfg["stop_pct"] * 0.75
    return cfg


def grid(base):
    for stop, rr, vm, ext, ts, mom, win in itertools.product(
            [0.008, 0.012, 0.016, 0.024], [2.0, 3.0], [2.0, 3.0], [0.02, 0.04], [30, 60, 999], [1, 0], [999, 120]):
        # keep money at risk per trade about constant: smaller size for wider stops
        px = round(min(1.5, 0.008 / stop), 2)
        yield with_params(base, stop_pct=stop, target_pct=round(stop * rr, 4), vol_mult=vm,
                          max_extension=ext, time_stop_min=ts, position_x=px,
                          momentum_exit=mom, entry_window_min=win)


def score_fn(s):
    """Rank settings: needs enough trades; reward profit, punish drawdown."""
    if s["trades"] < 30:
        return -1e9
    return s["pnl"] + 0.5 * s["max_drawdown"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--adopt", action="store_true")
    ap.add_argument("--inverse", action="store_true", help="test the 'do the opposite' idea")
    a = ap.parse_args()

    base = current_cfg()
    bars, days = load_bars(base["symbols"], a.days)
    split = int(len(days) * 2 / 3)
    train, test = days[:split], days[split:]
    lines = [f"# Scalper backtest {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
             f"{len(days)} trading days ({days[0]} to {days[-1]}), symbols {','.join(bars)}; "
             f"train {train[0]}..{train[-1]}, test {test[0]}..{test[-1]}. Costs: 0.03% slippage per side.", ""]
    cur = {k: base[k] for k in TUNABLE}
    res = {"current": {"params": cur, "all": stats(simulate(bars, days, base), days),
                       "train": stats(simulate(bars, train, base), train),
                       "test": stats(simulate(bars, test, base), test)}}
    lines += ["## Current settings", json.dumps(cur), "",
              f"- ALL : {res['current']['all']}", f"- TRAIN: {res['current']['train']}",
              f"- TEST : {res['current']['test']}", ""]

    if a.inverse:
        lines.append("## 'Do the opposite' test (signal on X -> buy the inverse fund; TSLA/NVDA skipped)")
        normal_pairs = with_params(base); normal_pairs["symbols"] = list(OPPOSITE)
        sub = {s: x for s, x in bars.items() if s in OPPOSITE}
        variants = [("NORMAL, same 4 funds (for comparison)", dict(base), False),
                    ("OPPOSITE, current rules", dict(base, inverse=1), True),
                    ("OPPOSITE, no momentum exit", dict(base, inverse=1, momentum_exit=0), True),
                    ("OPPOSITE, swap stop/target (-1.6% stop, +0.8% target)",
                     dict(base, inverse=1, stop_pct=base["target_pct"], target_pct=base["stop_pct"]), True)]
        for name, cfg, _ in variants:
            lines += [f"### {name}", f"- ALL  : {stats(simulate(sub, days, cfg), days)}",
                      f"- TRAIN: {stats(simulate(sub, train, cfg), train)}",
                      f"- TEST : {stats(simulate(sub, test, cfg), test)}", ""]

    if a.tune:
        results = []
        for cfg in grid(base):
            s = stats(simulate(bars, train, cfg), train)
            results.append((score_fn(s), s, cfg))
        results.sort(key=lambda r: r[0], reverse=True)
        lines.append("## Best 5 settings on TRAIN, then checked on TEST (never seen while tuning)")
        best = None
        for sc_, s_train, cfg in results[:5]:
            s_test = stats(simulate(bars, test, cfg), test)
            p = {k: cfg[k] for k in TUNABLE}
            lines += [json.dumps(p), f"- TRAIN: {s_train}", f"- TEST : {s_test}", ""]
            if best is None:
                best = (p, s_train, s_test, cfg)
        res["tuned"] = {"params": best[0], "train": best[1], "test": best[2],
                        "all": stats(simulate(bars, days, best[3]), days)}
        cur_test = res["current"]["test"]
        passed = (best[2]["pnl"] > 0 and best[2]["trades"] >= 5 and best[2]["pnl"] > cur_test["pnl"]
                  and best[1]["pnl"] > 0)
        res["adopted"] = bool(passed and a.adopt)
        verdict = ("ADOPTED: new settings profitable on unseen days and better than current."
                   if res["adopted"] else
                   "NOT adopted: " + ("would pass (run with --adopt)." if passed else
                                      "best settings did not make money on the unseen TEST days / not better than current."))
        lines += ["## Verdict", verdict]
        if res["adopted"]:
            os.makedirs(os.path.dirname(PARAMS_FILE), exist_ok=True)
            json.dump({"updated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
                       "params": best[0], "test": best[2], "train": best[1]}, open(PARAMS_FILE, "w"), indent=1)

    os.makedirs(REPORT_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    open(os.path.join(REPORT_DIR, f"{stamp}.md"), "w").write("\n".join(lines) + "\n")
    json.dump(res, open(os.path.join(REPORT_DIR, "latest.json"), "w"), indent=1, default=str)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
