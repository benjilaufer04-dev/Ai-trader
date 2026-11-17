"""Strategy lab: test many well-known trading styles on years of DAILY bars, honestly.

Every strategy gets a small set of settings. Settings are picked on TRAIN years only, then judged
on TEST years they never saw (walk-forward). A strategy "passes" only if it makes money on TEST,
beats buy-and-hold on risk (Sharpe) or drawdown there, and is not a fluke (enough trades, every
TEST year not a disaster). Report only - never trades, never changes the live bot.

Costs: 0.05% per side (slippage+spread on liquid ETFs). Signals use the close; trades happen at the
NEXT bar (open or close, as stated per strategy), so no look-ahead.

Usage: python research.py --data data/daily.csv.gz [--split 2024-01-01]
"""
import argparse
import itertools
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

COST = 0.0005
OUT_DIR = os.path.join(os.getenv("MEMORY_DIR", "memory"), "research")


# ---------- data ----------

def load(path):
    d = pd.read_csv(path)
    d["date"] = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    px = {f: d.pivot_table(index="date", columns="symbol", values=f) for f in ("open", "high", "low", "close")}
    return px


def rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, 1e-12))


# ---------- strategies: each returns daily strategy returns (pd.Series) + number of trades ----------
# Convention: pos[t] = fraction held from close t to close t+1, decided with data up to close t.

def hold_returns(px, sym, pos):
    """Close-to-close returns of holding `pos` (decided at close t, earned t->t+1) minus costs."""
    c = px["close"][sym].dropna()
    pos = pos.reindex(c.index).fillna(0)
    r = c.pct_change().shift(-1).fillna(0)
    turn = pos.diff().abs().fillna(pos.abs())
    trades = int((pos.diff().fillna(pos) > 0).sum())
    return (pos * r - turn * COST).shift(1).fillna(0), trades


def s_buyhold(px, sym):
    c = px["close"][sym].dropna()
    return hold_returns(px, sym, pd.Series(1.0, index=c.index))


def s_trend(px, sym, sma=200, signal=None):
    """Hold `sym` only while `signal` (default sym) closes above its N-day average. Else cash."""
    sig = px["close"][signal or sym].dropna()
    pos = (sig > sig.rolling(sma).mean()).astype(float)
    return hold_returns(px, sym, pos)


def s_rsi2(px, sym, entry=10, exit_sma=5, trend=200):
    """Connors RSI(2): buy a pullback in an uptrend, sell on the bounce."""
    c = px["close"][sym].dropna()
    r2 = rsi(c, 2)
    up = c > c.rolling(trend).mean()
    pos, held = [], 0.0
    ma = c.rolling(exit_sma).mean()
    for i in range(len(c)):
        if held and c.iloc[i] > ma.iloc[i]:
            held = 0.0
        elif not held and up.iloc[i] and r2.iloc[i] < entry:
            held = 1.0
        pos.append(held)
    return hold_returns(px, sym, pd.Series(pos, index=c.index))


def s_ibs(px, sym, thr=0.2, trend=0):
    """Internal bar strength: close near the day's low -> buy at close, sell next close."""
    c, h, l = (px[f][sym].dropna() for f in ("close", "high", "low"))
    ibs = (c - l) / (h - l).replace(0, np.nan)
    pos = (ibs < thr).astype(float)
    if trend:
        pos *= (c > c.rolling(trend).mean())
    return hold_returns(px, sym, pos)


def s_ibs_open(px, sym, thr=0.2):
    """Same IBS signal, but the pessimistic way: buy at the NEXT open (not the signal close), sell at
    that day's close. If the edge only exists when buying exactly at the close, this shows it."""
    o, c, h, l = (px[f][sym].dropna() for f in ("open", "close", "high", "low"))
    ibs = (c - l) / (h - l).replace(0, np.nan)
    sig = (ibs < thr).shift(1).fillna(False)
    r = ((c / o - 1 - 2 * COST) * sig).fillna(0)
    return r, int(sig.sum())


def deep_dive(px, split, thr=0.2):
    """IBS on every fund: is the pattern broad (real) or only in one lucky fund?"""
    lines = [f"## Deep dive: IBS (close in the bottom {int(thr * 100)}% of the day's range -> hold one day), every fund",
             "| fund | TRAIN Sharpe | TEST Sharpe | TEST CAGR | TEST DD | hold-the-fund TEST Sharpe | next-open version TEST Sharpe |",
             "|---|---|---|---|---|---|---|"]
    for sym in sorted(px["close"].columns):
        try:
            r, _ = s_ibs(px, sym, thr)
            ro, _ = s_ibs_open(px, sym, thr)
            bh, _ = s_buyhold(px, sym)
        except Exception:
            continue
        a, b = metrics(r[r.index < split]), metrics(r[r.index >= split])
        if "sharpe" not in a or "sharpe" not in b:
            continue
        lines.append(f"| {sym} | {a['sharpe']} | {b['sharpe']} | {b['cagr_%']}% | {b['max_dd_%']}% | "
                     f"{metrics(bh.loc[split:]).get('sharpe')} | {metrics(ro.loc[split:]).get('sharpe')} |")
    return lines + [""]


def near_close_check(minute_path, symbols=("SMH", "SOXL", "TQQQ", "QQQ"), thr=0.2, cut="15:55"):
    """Can we really trade IBS? The daily test buys AT the close, using the close to decide - not possible
    in practice. Here the bot decides with what it can see at 15:55 (today's high/low/price so far),
    buys at the 15:56 bar's open, and sells at the 15:56 open of the next day (or keeps holding if the
    signal fires again). Compared with the ideal close-to-close version on the same days."""
    m = pd.read_csv(minute_path)
    m["ts"] = pd.to_datetime(m["timestamp"], utc=True).dt.tz_convert("America/New_York")
    m["day"], m["hm"] = m["ts"].dt.date, m["ts"].dt.strftime("%H:%M")
    lines = [f"## Near-close check (IBS < {thr}, decide at {cut} with the minute bars, trade the next minute), "
             f"{m['day'].min()}..{m['day'].max()}",
             "| fund | days | trades | real-world total | ideal close-to-close total | real win rate | real avg/trade |",
             "|---|---|---|---|---|---|---|"]
    res = {}
    for sym in symbols:
        x = m[m["symbol"] == sym]
        if x.empty:
            continue
        rows = []
        for day, d in x.groupby("day"):
            seen = d[d["hm"] <= cut]
            nxt = d[d["hm"] > cut]
            if len(seen) < 300 or nxt.empty:
                continue
            h, l, p = seen["high"].max(), seen["low"].min(), seen["close"].iloc[-1]
            full_h, full_l, close = d["high"].max(), d["low"].min(), d["close"].iloc[-1]
            rows.append({"day": day, "ibs_live": (p - l) / (h - l) if h > l else 0.5, "fill": nxt["open"].iloc[0],
                         "ibs_ideal": (close - full_l) / (full_h - full_l) if full_h > full_l else 0.5, "close": close})
        df = pd.DataFrame(rows)
        if len(df) < 20:
            continue
        real, ideal = [], []
        for i in range(len(df) - 1):
            if df["ibs_live"].iloc[i] < thr:
                real.append(df["fill"].iloc[i + 1] / df["fill"].iloc[i] - 1 - 2 * COST)
            if df["ibs_ideal"].iloc[i] < thr:
                ideal.append(df["close"].iloc[i + 1] / df["close"].iloc[i] - 1 - 2 * COST)
        tot = lambda r: round((float(np.prod([1 + v for v in r])) - 1) * 100, 1) if r else 0.0
        res[sym] = {"days": len(df), "trades": len(real), "real_total_%": tot(real), "ideal_total_%": tot(ideal),
                    "real_win_rate": round(float(np.mean([v > 0 for v in real])), 2) if real else 0,
                    "real_avg_%": round(float(np.mean(real)) * 100, 2) if real else 0}
        r = res[sym]
        lines.append(f"| {sym} | {r['days']} | {r['trades']} | {r['real_total_%']}% | {r['ideal_total_%']}% | "
                     f"{r['real_win_rate']} | {r['real_avg_%']}% |")
    return lines + [""], res


def s_overnight(px, sym):
    """Buy at the close, sell at the next open (the 'overnight' effect)."""
    o, c = px["open"][sym].dropna(), px["close"][sym].dropna()
    r = (o.shift(-1) / c - 1 - 2 * COST).shift(1).fillna(0)
    return r, len(r)


def s_rotation(px, universe, lookback=126, top=3, trend=200):
    """Monthly: hold the `top` strongest funds of `universe` by past return, only those above their
    N-day average (else that slot sits in cash)."""
    c = px["close"][universe].dropna(how="all")
    mom = c / c.shift(lookback) - 1
    ok = c > c.rolling(trend).mean()
    month_end = c.index.to_series().groupby(c.index.to_period("M")).max().values
    w = pd.DataFrame(0.0, index=c.index, columns=universe)
    cur = pd.Series(0.0, index=universe)
    trades = 0
    for t in c.index:
        if t in month_end:
            m = mom.loc[t].where(ok.loc[t]).dropna().sort_values(ascending=False)
            new = pd.Series(0.0, index=universe)
            for s in m.index[:top]:
                if m[s] > 0:
                    new[s] = 1.0 / top
            trades += int(((new > 0) & (cur == 0)).sum())
            cur = new
        w.loc[t] = cur
    r = c.pct_change().shift(-1).fillna(0)
    turn = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    return ((w * r).sum(axis=1) - turn * COST).shift(1).fillna(0), trades


# ---------- scoring ----------

def metrics(r, trades=None):
    r = r.dropna()
    if len(r) < 20:
        return {"days": len(r)}
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1
    dd = float((eq / eq.cummax() - 1).min())
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0
    by_year = {str(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r.groupby(r.index.year)}
    out = {"cagr_%": round(float(cagr) * 100, 1), "total_%": round(float(eq.iloc[-1] - 1) * 100, 1),
           "max_dd_%": round(dd * 100, 1), "sharpe": round(sharpe, 2),
           "exposure_%": round(float((r != 0).mean()) * 100, 0), "by_year_%": by_year}
    if trades is not None:
        out["trades"] = trades
    return out


def candidates(px):
    have = set(px["close"].columns)
    sym = [s for s in ("SPY", "QQQ", "IWM", "SMH", "TQQQ", "SOXL", "UPRO", "TNA") if s in have]
    yield "buy&hold", {"sym": sym}, s_buyhold
    yield "trend (above N-day avg)", {"sym": sym, "sma": [50, 100, 200]}, s_trend
    lev = [(l, b) for l, b in (("TQQQ", "QQQ"), ("UPRO", "SPY"), ("SOXL", "SMH"), ("TNA", "IWM")) if l in have and b in have]
    yield "leveraged fund, trend of its index", {"pair": lev, "sma": [50, 100, 200]}, \
        lambda p, pair, sma: s_trend(p, pair[0], sma, signal=pair[1])
    yield "RSI(2) pullback", {"sym": sym, "entry": [5, 10, 20], "exit_sma": [5, 10], "trend": [100, 200]}, s_rsi2
    yield "IBS close-near-low", {"sym": sym, "thr": [0.1, 0.2, 0.3], "trend": [0, 200]}, s_ibs
    yield "overnight hold", {"sym": sym}, s_overnight
    sectors = [s for s in ("XLE", "XLF", "XLK", "XLV", "XLY", "XLP", "XLI", "XLU", "SMH", "GLD", "TLT") if s in have]
    yield "sector rotation", {"universe": [tuple(sectors)], "lookback": [63, 126, 252], "top": [1, 2, 3]}, \
        lambda p, universe, lookback, top: s_rotation(p, list(universe), lookback, top)


def shuffled(px, seed):
    """Placebo market: the same days (all funds together, so correlations stay), in random order.
    Drift and volatility survive, but any real pattern in the ORDER of days is destroyed. A strategy
    whose real result is no better than on placebo markets has no edge - it was luck."""
    c = px["close"]
    rel = {f: (px[f] / c.shift(1)) for f in ("open", "high", "low")}
    ret = (c / c.shift(1)).iloc[1:].fillna(1.0)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ret))
    newc = ret.iloc[idx].cumprod()
    newc.index = ret.index
    newc = newc * c.ffill().bfill().iloc[0]
    newc = newc.where(c.iloc[1:].notna())
    out = {"close": newc}
    for f in ("open", "high", "low"):
        r = rel[f].iloc[1:].iloc[idx]
        r.index = ret.index
        out[f] = newc.shift(1) * r
    return out


def evaluate(px, split, lines=None):
    """Pick each strategy's settings on TRAIN, score on TEST. Returns {strategy: result}."""
    out = {}
    spy = metrics(s_buyhold(px, "SPY")[0].loc[split:]) if "SPY" in px["close"] else {}
    for name, grid, fn in candidates(px):
        keys = list(grid)
        rows = []
        for combo in itertools.product(*[grid[k] for k in keys]):
            kw = dict(zip(keys, combo))
            try:
                r, n = fn(px, **kw)
            except Exception as e:
                print(f"skip {name} {kw}: {e}")
                continue
            mtr, mte = metrics(r[r.index < split]), metrics(r[r.index >= split])
            if "sharpe" not in mtr or "sharpe" not in mte:
                continue
            rows.append((mtr["sharpe"], kw, mtr, mte, n))
        if not rows:
            continue
        rows.sort(key=lambda x: x[0], reverse=True)
        _, kw, mtr, mte, n = rows[0]
        # robustness: the 10 best TRAIN settings, as a group, must still do well on TEST
        robust = float(np.median([x[3]["sharpe"] for x in rows[:10]]))
        # a timing rule must beat simply holding the same fund over the TEST years (risk-adjusted)
        held = kw.get("sym") or (kw.get("pair") or [None])[0]
        own = metrics(s_buyhold(px, held)[0].loc[split:])["sharpe"] if held and name != "buy&hold" else -9
        rules = (mte["total_%"] > 0 and mtr["total_%"] > 0 and mte["sharpe"] >= max(0.8, spy.get("sharpe", 0))
                 and robust >= 0.6 and mte["sharpe"] > own and mte["max_dd_%"] > -35
                 and min(mte["by_year_%"].values()) > -20)
        out[name] = {"strategy": name, "params": kw, "train": mtr, "test": mte, "robust_test_sharpe": robust,
                     "own_buyhold_test_sharpe": own, "rules_ok": bool(rules), "trades": n}
        if lines is not None:
            lines.append(f"## {name}: best 3 settings on TRAIN, then TEST")
            for _, k2, a2, b2, n2 in rows[:3]:
                lines += [f"`{json.dumps(k2, default=str)}` trades(all)={n2}", f"- TRAIN {a2}", f"- TEST  {b2}"]
            lines += [f"- top-10 TRAIN settings, median TEST Sharpe: {robust:.2f}; holding the same fund: {own}", ""]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/daily.csv.gz")
    ap.add_argument("--split", default="2024-01-01", help="first TEST day; everything before is TRAIN")
    ap.add_argument("--placebos", type=int, default=20)
    ap.add_argument("--minute", default="data/minute.csv.gz", help="minute bars for the near-close check")
    a = ap.parse_args()
    px = load(a.data)
    split = pd.Timestamp(a.split)
    first, last = px["close"].index.min().date(), px["close"].index.max().date()
    lines = [f"# Strategy lab {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
             f"Daily bars {first}..{last}. TRAIN before {split.date()}, TEST from {split.date()} (never used to pick settings).",
             f"Costs {COST * 100:.2f}% per side. Signals at the close, trade next bar. Report only.",
             f"Luck check: the whole search is repeated on {a.placebos} placebo markets (same days, shuffled order). "
             "PASS needs the rules below AND a real TEST Sharpe better than 90% of placebo runs.", ""]
    bench = {}
    for b in ("SPY", "QQQ"):
        if b in px["close"]:
            r, _ = s_buyhold(px, b)
            bench[b] = {"train": metrics(r[r.index < split]), "test": metrics(r[r.index >= split])}
    lines += ["## Benchmarks (just hold)"] + [f"- {b} TRAIN {v['train']}\n- {b} TEST  {v['test']}" for b, v in bench.items()] + [""]
    real = evaluate(px, split, lines)
    lines += deep_dive(px, split)
    near = {}
    if a.minute and os.path.exists(a.minute):
        nl, near = near_close_check(a.minute)
        lines += nl
    placebo = {k: [] for k in real}
    for i in range(a.placebos):
        for k, v in evaluate(shuffled(px, i), split).items():
            placebo.setdefault(k, []).append(v["test"]["sharpe"])
    summary = []
    for k, v in real.items():
        pl = placebo.get(k, [])
        v["luck_p"] = round(float(np.mean([x >= v["test"]["sharpe"] for x in pl])), 2) if pl else None
        v["placebo_median_sharpe"] = round(float(np.median(pl)), 2) if pl else None
        v["passed"] = bool(v["rules_ok"] and v["luck_p"] is not None and v["luck_p"] <= 0.10)
        summary.append(v)
    lines.append("## Ranking on TEST (best TRAIN pick of each strategy)")
    lines.append("PASS = TEST Sharpe >= max(0.8, SPY), both halves positive, top-10 median TEST Sharpe >= 0.6, "
                 "beats holding the same fund, DD > -35%, no TEST year < -20%, luck p <= 0.10")
    for s in sorted(summary, key=lambda s: s["test"]["sharpe"], reverse=True):
        lines.append(f"- {'PASS' if s['passed'] else 'fail'} {s['strategy']} {json.dumps(s['params'], default=str)}: "
                     f"TEST CAGR {s['test']['cagr_%']}% DD {s['test']['max_dd_%']}% Sharpe {s['test']['sharpe']} | "
                     f"TRAIN CAGR {s['train']['cagr_%']}% DD {s['train']['max_dd_%']}% Sharpe {s['train']['sharpe']} | "
                     f"robust {s['robust_test_sharpe']:.2f} | luck p {s['luck_p']} (placebo median {s['placebo_median_sharpe']})")
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    open(os.path.join(OUT_DIR, f"daily_{stamp}.md"), "w").write("\n".join(lines) + "\n")
    json.dump({"benchmarks": bench, "strategies": summary, "near_close": near}, open(os.path.join(OUT_DIR, "daily_latest.json"), "w"),
              indent=1, default=str)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
