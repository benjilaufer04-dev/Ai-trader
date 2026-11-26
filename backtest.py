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
            "ts": df["timestamp"].values, "ext": ext.fillna(0).values, "vwap": ind["vwap"].values,
            "daymove": (c / o.iloc[0] - 1).values}


def market_state(q, m, inverse_fund):
    """Was the trade WITH the whole market (QQQ) at minute m? 'with' / 'against' / 'mixed'."""
    if q is None:
        return "n/a"
    i = np.searchsorted(q["minute"], m)
    if i >= len(q["minute"]) or q["minute"][i] != m:
        return "n/a"
    up = q["c"][i] > q["vwap"][i] and q["c"][i] > q["o"][0]
    down = q["c"][i] < q["vwap"][i] and q["c"][i] < q["o"][0]
    if not (up or down):
        return "mixed"
    return "with" if up != bool(inverse_fund) else "against"


INVERSE_FUNDS = {"SQQQ", "SOXS", "SPXU", "TZA"}


def simulate(bars, days, cfg, equity=100_000, slip=0.0003, ctx=None):
    """Event-driven over all symbols, minute by minute. Returns list of trades.
    ctx: optional QQQ bars, only used to label each trade (with/against the market) for learning."""
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
        q = None
        if ctx is not None:
            dq = ctx[ctx["day"] == day]
            q = prep_day(dq.drop(columns="day"), cfg) if len(dq) > 30 else None
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
                                   "why": why, "minutes": m - t["m"], "entry_min": t["m"],
                                   **t.get("ctx", {})})
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
                    cands.append((p["score"][idx], s2, p2["o"][j], {}))
                    continue
                feat = {"vol_x": float(p["score"][idx]), "ext": float(p["ext"][idx]),
                        "daymove": float(p["daymove"][idx]), "market": market_state(q, m - 1, s in INVERSE_FUNDS),
                        "weekday": pd.Timestamp(day).day_name()[:3]}
                cands.append((p["score"][idx], s, p["o"][idx + 1], feat))
            for score, s, px, feat in sorted(cands, key=lambda x: x[0], reverse=True):
                if len(open_pos) >= cfg["max_positions"]:
                    break
                entry = px * (1 + slip)
                open_pos[s] = {"entry": entry, "stop": entry * (1 - cfg["stop_pct"]),
                               "target": entry * (1 + cfg["target_pct"]), "best": entry, "m": m,
                               "usd": equity * cfg["position_x"], "ctx": feat}
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


def breakdown(trades):
    """P/L split by symbol and by entry time (30-min buckets after the open), to find filters
    that remove losers on BOTH train and test before changing the live bot."""
    if not trades:
        return {"by_symbol": {}, "by_entry_time": {}}
    df = pd.DataFrame(trades)
    df["bucket"] = (df["entry_min"] // 30 * 30).astype(int).map(lambda b: f"{b:03d}-{b + 29:03d}m")

    def agg(col):
        out = {}
        for k, g in df.groupby(col):
            w, l = g[g.pnl > 0].pnl.sum(), -g[g.pnl <= 0].pnl.sum()
            out[str(k)] = {"n": int(len(g)), "pnl": round(float(g.pnl.sum()), 0),
                           "pf": round(float(w / l), 2) if l > 0 else 99}
        return out
    return {"by_symbol": agg("symbol"), "by_entry_time": agg("bucket")}


FEATURES = {
    "signal volume": ("vol_x", [(0, 3, "<3x"), (3, 5, "3-5x"), (5, 1e9, "5x+")]),
    "distance above VWAP": ("ext", [(-1, 0.005, "<0.5%"), (0.005, 0.01, "0.5-1%"), (0.01, 1, "1-2%")]),
    "move since the open": ("daymove", [(-1, 0, "down"), (0, 0.02, "0-2%"), (0.02, 0.05, "2-5%"), (0.05, 9, "5%+")]),
    "entry time": ("entry_min", [(0, 30, "first 30m"), (30, 60, "30-60m"), (60, 120, "60-120m"), (120, 999, "later")]),
}


def label(trade, feature):
    key, buckets = FEATURES[feature]
    v = trade.get(key)
    if v is None:
        return None
    for lo, hi, name in buckets:
        if lo <= v < hi:
            return name
    return None


def patterns(train_trades, test_trades, min_n=8):
    """The learning step: for every trade condition (market direction, signal strength, time...),
    compare results on TRAIN and TEST. A pattern counts only if it shows up on BOTH halves with
    enough trades: 'AVOID' if it lost on both (PF < 0.8), 'FAVOR' if it won on both (PF > 1.5)."""
    def table(trades):
        out = {}
        for t in trades:
            labs = {f: label(t, f) for f in FEATURES}
            labs["market (QQQ)"] = t.get("market")
            labs["weekday"] = t.get("weekday")
            for f, b in labs.items():
                if b is None or b == "n/a":
                    continue
                d = out.setdefault((f, b), {"n": 0, "win": 0.0, "loss": 0.0})
                d["n"] += 1
                d["win" if t["pnl"] > 0 else "loss"] += abs(t["pnl"])
        return out
    tr, te = table(train_trades), table(test_trades)

    def overall(trades):
        w = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        l = -sum(t["pnl"] for t in trades if t["pnl"] <= 0)
        return w / l if l > 0 else 99
    o_tr, o_te = overall(train_trades), overall(test_trades)
    rows = []
    for k in sorted(set(tr) | set(te)):
        a, b = tr.get(k, {"n": 0, "win": 0, "loss": 0}), te.get(k, {"n": 0, "win": 0, "loss": 0})
        pf = lambda d: round(d["win"] / d["loss"], 2) if d["loss"] > 0 else 99
        flag = ""
        # weekday "effects" are almost always noise -> shown, never flagged
        if a["n"] >= min_n and b["n"] >= min_n and k[0] != "weekday":
            if pf(a) < min(0.8, o_tr) and pf(b) < min(0.8, o_te):
                flag = "AVOID"
            elif pf(a) > max(1.5, 1.25 * o_tr) and pf(b) > max(1.5, 1.25 * o_te):
                flag = "FAVOR"
        rows.append({"feature": k[0], "bucket": k[1], "train_n": a["n"], "train_pf": pf(a),
                     "train_pnl": round(a["win"] - a["loss"]), "test_n": b["n"], "test_pf": pf(b),
                     "test_pnl": round(b["win"] - b["loss"]), "flag": flag})
    return rows


def live_trades_with_context(bars, ctx, path=None):
    """Re-create the entry conditions of every LIVE trade from the same bars the backtest uses,
    so live and backtest trades are judged by the same yardstick."""
    path = path or os.path.join(os.getenv("MEMORY_DIR", "memory"), "scalper_trades.jsonl")
    if not os.path.exists(path):
        return []
    rows = [json.loads(l) for l in open(path) if l.strip()]
    buys = {}
    out = []
    for r in rows:
        if r.get("side") == "buy":
            buys[r["symbol"]] = r
        elif r.get("side") == "sell" and r["symbol"] in buys:
            b = buys.pop(r["symbol"])
            t = {"day": r["ts"][:10], "symbol": r["symbol"], "pnl": r.get("pnl", 0), "why": r.get("reason", "")}
            ts = pd.Timestamp(b["ts"]).tz_convert("America/New_York")
            x = bars.get(r["symbol"])
            if x is not None:
                d = x[x["day"] == ts.date()]
                if len(d) > 30:
                    p = prep_day(d.drop(columns="day"), sc.CFG)
                    m = ts.hour * 60 + ts.minute - 570 - 1      # signal bar = the minute before entry
                    i = np.searchsorted(p["minute"], m)
                    if i < len(p["minute"]):
                        q = None
                        if ctx is not None:
                            dq = ctx[ctx["day"] == ts.date()]
                            q = prep_day(dq.drop(columns="day"), sc.CFG) if len(dq) > 30 else None
                        t.update({"entry_min": m + 1, "vol_x": b.get("signal_volume_x"), "ext": float(p["ext"][i]),
                                  "daymove": float(p["daymove"][i]), "weekday": ts.day_name()[:3],
                                  "market": market_state(q, m, r["symbol"] in INVERSE_FUNDS)})
            out.append(t)
    return out


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


def should_adopt(train, test, cur_test, equity=100_000):
    """Auto-adopt only settings that meet goal.md criterion 1 on the unseen TEST days
    (profit > 0, profit factor >= 1.3), keep TEST drawdown under 5% of the account,
    make money on TRAIN too, and beat the current settings on TEST."""
    return (test["trades"] >= 5 and test["pnl"] > 0 and test["pnl"] > cur_test["pnl"]
            and test["profit_factor"] >= 1.3 and test["max_drawdown"] > -0.05 * equity
            and train["pnl"] > 0)


def older_ok(new_old, cur_old):
    """Second gate (reviewer 11-25): the new settings must not be worse than the current ones on
    the OLDER days before the 60-day window (days neither tuning nor TEST used). On 11-24 a tuned
    setting passed TEST but lost Jun-Sep (PF 0.82, DD 13.7%) where the current one was ~flat:
    it fit only the recent market. Require: profit not lower, drawdown not >25% deeper."""
    if new_old is None or cur_old is None:
        return True  # no older data loaded: gate not applied
    return (new_old["pnl"] >= cur_old["pnl"]
            and new_old["max_drawdown"] >= 1.25 * min(cur_old["max_drawdown"], -1.0))


# Candidate rule changes from the Lessons / reviewer notes, re-checked every night (report only).
# Each is the current settings plus one change; it is only a candidate for going live if it beats
# the current settings on TRAIN, TEST and the OLDER days, with TEST PF >= 1.3 (see rule_check_passes).
RULE_CHECKS = [
    ("skip the first 30 min (entries from 9:30+30m; lessons: first-30m entries lose)", {"skip_open_min": 30}),
]


def rule_check_passes(var, cur):
    """var/cur: {"train","test","older"} -> stats. Better profit on every part loaded, TEST PF >= 1.3."""
    parts = [k for k in ("train", "test", "older") if var.get(k) is not None and cur.get(k) is not None]
    return (bool(parts) and all(var[k]["pnl"] > cur[k]["pnl"] for k in parts)
            and var["test"]["profit_factor"] >= 1.3)


def adoption_frozen(path=None, today=None):
    """A reviewer can pause auto-adoption by putting "freeze_until": "YYYY-MM-DD" (and a
    "freeze_reason") in scalper_params.json. Returns the date string while frozen, else None."""
    path = path or PARAMS_FILE
    today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        until = json.load(open(path)).get("freeze_until")
    except Exception:
        return None
    return until if until and today <= until else None


def _short(s):
    return f"{s['trades']} tr ${s['pnl']:+,.0f} PF {s['profit_factor']} DD ${s['max_drawdown']:,.0f}"


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
    ap.add_argument("--older", type=int, default=0,
                    help="with --tune: also load this many OLDER days and require new settings not worse there")
    ap.add_argument("--inverse", action="store_true", help="test the 'do the opposite' idea")
    a = ap.parse_args()

    base = current_cfg()
    n_older = a.older if a.tune else 0
    bars, days = load_bars(list(dict.fromkeys(base["symbols"] + ["QQQ"])), a.days + n_older)
    older, days = (days[:-a.days], days[-a.days:]) if n_older else ([], days)
    ctx = bars["QQQ"] if "QQQ" in bars else None
    if "QQQ" not in base["symbols"]:
        bars.pop("QQQ", None)
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
    tr_trades, te_trades = simulate(bars, train, base, ctx=ctx), simulate(bars, test, base, ctx=ctx)
    bd_train, bd_test = breakdown(tr_trades), breakdown(te_trades)
    res["current"]["breakdown"] = {"train": bd_train, "test": bd_test}
    lines += ["## Current settings: where the P/L comes from (look for losers on BOTH train and test)"]
    for part in ("by_symbol", "by_entry_time"):
        lines.append(f"- {part} TRAIN: {bd_train[part]}")
        lines.append(f"- {part} TEST : {bd_test[part]}")
    lines.append("")

    # ---- learning: which conditions keep winning / losing on BOTH halves, and the live trades ----
    pats = patterns(tr_trades, te_trades)
    res["patterns"] = pats
    lines += ["## Lessons: conditions that win or lose on BOTH train and test (min 8 trades each)",
              "FAVOR = PF > 1.5 and clearly above the strategy's overall PF on both halves; AVOID = PF < 0.8 on both. "
              "Only these are candidates for a rule change (then tested as a filter on train AND test).", ""]
    for r in pats:
        if r["flag"]:
            lines.append(f"- **{r['flag']}** {r['feature']} = {r['bucket']}: TRAIN {r['train_n']} tr PF {r['train_pf']} "
                         f"(${r['train_pnl']:+,}) | TEST {r['test_n']} tr PF {r['test_pf']} (${r['test_pnl']:+,})")
    if not any(r["flag"] for r in pats):
        lines.append("- none yet")
    lines += ["", "<details><summary>All conditions</summary>", ""]
    lines += [f"- {r['feature']} = {r['bucket']}: TRAIN {r['train_n']} PF {r['train_pf']} | TEST {r['test_n']} PF {r['test_pf']}"
              for r in pats]
    lines += ["", "</details>", ""]
    live = live_trades_with_context(bars, ctx)
    flagged = {(r["feature"], r["bucket"]): r["flag"] for r in pats if r["flag"]}
    if live:
        lines.append("## Live trades, judged by the same conditions")
        for t in live:
            labs = {f: label(t, f) for f in FEATURES}
            labs["market (QQQ)"] = t.get("market")
            notes = [f"{flagged[(f, b)]}:{f}={b}" for f, b in labs.items() if (f, b) in flagged]
            lines.append(f"- {t['day']} {t['symbol']} ${t['pnl']:+,.0f} ({t['why'][:30]}) | market {t.get('market')}, "
                         f"vol x{t.get('vol_x')}, {labs['entry time']}, day move {labs['move since the open']}"
                         + (f" | {', '.join(notes)}" if notes else ""))
        lines.append("")

    if n_older and older:
        lines += ["## Rule checks (one change vs current; must beat current on TRAIN, TEST and OLDER "
                  f"{older[0]}..{older[-1]}; report only)", ""]
        cur_parts = {"train": res["current"]["train"], "test": res["current"]["test"],
                     "older": stats(simulate(bars, older, base), older)}
        lines.append(f"- CURRENT: TRAIN {_short(cur_parts['train'])} | TEST {_short(cur_parts['test'])} | "
                     f"OLDER {_short(cur_parts['older'])}")
        res["rule_checks"] = []
        for name, change in RULE_CHECKS:
            cfg = dict(base, **change)
            var = {"train": stats(simulate(bars, train, cfg), train), "test": stats(simulate(bars, test, cfg), test),
                   "older": stats(simulate(bars, older, cfg), older)}
            ok = rule_check_passes(var, cur_parts)
            res["rule_checks"].append({"name": name, "change": change, **var, "passes": ok})
            lines.append(f"- {name}: TRAIN {_short(var['train'])} | TEST {_short(var['test'])} | "
                         f"OLDER {_short(var['older'])} -> {'PASSES (candidate)' if ok else 'no'}")
        lines.append("")

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
        passed = should_adopt(best[1], best[2], cur_test)
        if passed and older:
            new_old = stats(simulate(bars, older, best[3]), older)
            cur_old = stats(simulate(bars, older, base), older)
            ok = older_ok(new_old, cur_old)
            res["older_check"] = {"days": f"{older[0]}..{older[-1]}", "new": new_old, "current": cur_old, "ok": ok}
            lines += [f"## Older-days check ({len(older)} days {older[0]}..{older[-1]}, never used above)",
                      f"- NEW    : {new_old}", f"- CURRENT: {cur_old}",
                      f"- {'OK: not worse on older days' if ok else 'FAIL: worse on older days (fits only the recent market)'}", ""]
            passed = ok
        frozen = adoption_frozen()
        res["adopted"] = bool(passed and a.adopt and not frozen)
        verdict = (f"NOT adopted: auto-adoption paused by the reviewer until {frozen}"
                   f" ({'would pass' if passed else 'would not pass'})." if frozen else
                   "ADOPTED: new settings profitable on unseen days and better than current."
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
