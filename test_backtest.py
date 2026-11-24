"""Smoke tests for the backtest learning step and the strategy lab (synthetic data, no network)."""
import json

import numpy as np
import pandas as pd

import backtest as bt
import research as rs


def fake_minute_bars(symbols, days=6, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for s in symbols:
        frames = []
        for d in pd.bdate_range("2026-10-01", periods=days):
            idx = pd.date_range(d.tz_localize("America/New_York") + pd.Timedelta("9h30min"), periods=390,
                                freq="1min").tz_convert("UTC")
            c = 50 * np.exp(np.cumsum(rng.normal(0.0002, 0.002, 390)))
            o = np.r_[c[0], c[:-1]]
            v = rng.integers(1000, 5000, 390).astype(float)
            v[rng.integers(30, 380, 15)] *= 8          # some volume spikes -> signals
            x = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.001, "low": np.minimum(o, c) * 0.999,
                              "close": c, "volume": v}, index=pd.Index(idx, name="timestamp"))
            x["day"] = x.index.tz_convert("America/New_York").date
            frames.append(x)
        out[s] = pd.concat(frames)
    days = sorted({d for x in out.values() for d in x["day"].unique()})
    return out, days


def test_simulate_labels_trades_and_patterns_run():
    bars, days = fake_minute_bars(["TQQQ", "SQQQ", "QQQ"])
    ctx = bars.pop("QQQ")
    cfg = bt.with_params(dict(bt.sc.CFG), vol_mult=2.0, entry_window_min=999)
    trades = bt.simulate(bars, days, cfg, ctx=ctx)
    assert trades, "synthetic volume spikes should create trades"
    assert all(t["market"] in ("with", "against", "mixed", "n/a") for t in trades)
    assert all("vol_x" in t and "entry_min" in t for t in trades)
    rows = bt.patterns(trades[: len(trades) // 2], trades[len(trades) // 2:], min_n=1)
    assert rows and all(r["flag"] in ("", "AVOID", "FAVOR") for r in rows)


def test_patterns_flags_only_when_both_halves_agree():
    lose = [{"pnl": -100, "entry_min": 10, "market": "against"} for _ in range(8)]
    win = [{"pnl": 100, "entry_min": 10, "market": "with"} for _ in range(8)]
    rows = {(r["feature"], r["bucket"]): r["flag"] for r in bt.patterns(lose + win, lose + win)}
    assert rows[("market (QQQ)", "against")] == "AVOID"
    assert rows[("market (QQQ)", "with")] == "FAVOR"
    rows = {(r["feature"], r["bucket"]): r["flag"] for r in bt.patterns(lose, win)}
    assert rows[("market (QQQ)", "against")] == ""     # lost on train only -> not a lesson


def test_live_trades_get_same_context(tmp_path):
    bars, days = fake_minute_bars(["TQQQ", "QQQ"])
    ctx = bars.pop("QQQ")
    ts = pd.Timestamp(f"{days[2]} 10:30", tz="America/New_York").tz_convert("UTC")
    log = tmp_path / "t.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in [
        {"ts": ts.isoformat(), "symbol": "TQQQ", "side": "buy", "qty": 1, "entry": 50, "signal_volume_x": 3.1},
        {"ts": (ts + pd.Timedelta("20min")).isoformat(), "symbol": "TQQQ", "side": "sell", "pnl": -5, "reason": "stop"}]))
    live = bt.live_trades_with_context(bars, ctx, str(log))
    assert len(live) == 1 and live[0]["entry_min"] == 60 and live[0]["market"] in ("with", "against", "mixed")


def test_research_lab_runs_and_placebo_destroys_order():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-02", "2025-12-31", tz="America/New_York")
    rows = []
    for s in ("SPY", "QQQ", "TQQQ"):
        c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(dates))))
        rows.append(pd.DataFrame({"symbol": s, "timestamp": dates.tz_convert("UTC"), "open": c, "high": c * 1.01,
                                  "low": c * 0.99, "close": c, "volume": 1}))
    px = {f: pd.concat(rows).assign(date=lambda d: pd.to_datetime(d["timestamp"]).dt.tz_convert("America/New_York")
                                    .dt.normalize().dt.tz_localize(None)).pivot_table(index="date", columns="symbol",
                                                                                     values=f)
          for f in ("open", "high", "low", "close")}
    res = rs.evaluate(px, pd.Timestamp("2024-01-01"))
    assert "RSI(2) pullback" in res and "buy&hold" in res
    sh = rs.shuffled(px, 0)
    # same set of daily moves, different order
    a = np.sort((px["close"]["SPY"].pct_change().dropna()).round(10).values)
    b = np.sort((sh["close"]["SPY"].pct_change().dropna()).round(10).values)
    assert np.allclose(a[:50], b[:50]) and not np.allclose(px["close"]["SPY"].values[1:], sh["close"]["SPY"].values)


def test_near_close_check_runs(tmp_path):
    bars, days = fake_minute_bars(["SMH"], days=30)
    df = bars["SMH"].drop(columns="day").reset_index()
    df["symbol"] = "SMH"
    path = tmp_path / "m.csv.gz"
    df.to_csv(path, index=False)
    lines, res = rs.near_close_check(str(path), symbols=("SMH",), thr=0.5)
    assert "SMH" in res and res["SMH"]["days"] == 30 and res["SMH"]["trades"] > 0


def _day(o, h, l, c, post_open, post_low=None):
    """One day: a single 'pre' bar summarising the day up to 15:55, then one post bar."""
    pl = post_open if post_low is None else post_low
    return {"pre": [(o, h, l, c)], "post": [(post_open, post_open, pl, post_open)]}


def test_sim_swing_matches_live_rules():
    # day 0 closes weak -> buy at 100; day 1 bounces (IBS high) -> sell at 104
    days = [_day(110, 110, 99, 100, 100), _day(101, 105, 100, 104.5, 104)]
    t, why = rs.sim_swing(days, 0.2, 0.06)
    assert why == ["bounce"] and abs(t[0] - (0.04 - 2 * rs.COST)) < 1e-9


def test_sim_swing_stop_then_rebuy_same_day():
    # buy at 100 (stop 94); next day crashes to 88 (stop fills at 94), closes weak -> buys again at 88.5
    days = [_day(110, 110, 99, 100, 100), _day(99, 99, 88, 88.4, 88.5), _day(89, 93, 88.5, 92.9, 93)]
    t, why = rs.sim_swing(days, 0.2, 0.06)
    assert why == ["stop", "bounce"]
    assert abs(t[0] - (-0.06 - 2 * rs.COST)) < 1e-9 and t[1] > 0.04
    # same days without a stop: one trade held through the drop, sold on the bounce
    t2, why2 = rs.sim_swing(days, 0.2, 0)
    assert why2 == ["bounce"]
    assert len(t2) == 1 and t2[0] < 0


def test_sim_swing_gap_below_stop_fills_at_open_and_max_days():
    days = [_day(110, 110, 99, 100, 100), _day(90, 91, 89, 89.5, 89.5)]
    t, why = rs.sim_swing(days, 0.2, 0.06)
    assert why == ["stop"] and abs(t[0] - (-0.10 - 2 * rs.COST)) < 1e-9   # gapped to 90, not 94
    weak = [_day(110, 110, 99, 100, 100)] + [_day(100, 100, 99, 99.1, 99.1) for _ in range(6)]
    t, why = rs.sim_swing(weak, 0.2, 0)
    assert why[0] == "max days"


def test_swing_rules_check_runs(tmp_path):
    bars, _ = fake_minute_bars(["SMH"], days=30)
    rows = []
    for s, df in bars.items():
        d = df.drop(columns="day").reset_index()
        d["symbol"] = s
        rows.append(d)
    p = tmp_path / "m.csv"
    pd.concat(rows).to_csv(p, index=False)
    lines, res = rs.swing_rules_check(str(p), thr=0.5)
    assert any("stop" in l for l in lines) and "SMH stop 0.06" in res


def test_adoption_frozen(tmp_path):
    import json
    import backtest
    p = tmp_path / "params.json"
    p.write_text(json.dumps({"params": {}, "freeze_until": "2026-12-01"}))
    assert backtest.adoption_frozen(str(p), "2026-11-25") == "2026-12-01"
    assert backtest.adoption_frozen(str(p), "2026-12-01") == "2026-12-01"
    assert backtest.adoption_frozen(str(p), "2026-12-02") is None
    p.write_text(json.dumps({"params": {}}))
    assert backtest.adoption_frozen(str(p), "2026-11-25") is None
    assert backtest.adoption_frozen(str(tmp_path / "missing.json"), "2026-11-25") is None
