"""Scalper rules, no network."""
from datetime import datetime, timedelta, timezone

import pandas as pd

import scalper as sc


def bars(closes, vols=None, opens=None):
    n = len(closes)
    vols = vols or [1000] * n
    opens = opens or [c - 0.05 for c in closes]
    return pd.DataFrame({"open": opens, "high": [c + 0.02 for c in closes], "low": [c - 0.1 for c in closes],
                         "close": closes, "volume": vols})


def test_breakout_on_volume_is_a_buy():
    closes = [100 + i * 0.02 for i in range(30)] + [101.5]
    vols = [1000] * 30 + [3000]
    assert sc.entry_signal(bars(closes, vols)) > 1.3


def test_no_buy_without_volume_or_in_downtrend():
    closes = [100 + i * 0.02 for i in range(30)] + [101.5]
    assert sc.entry_signal(bars(closes)) == 0                       # normal volume
    down = [110 - i * 0.2 for i in range(30)] + [104.5]
    assert sc.entry_signal(bars(down, [1000] * 30 + [3000])) == 0   # below VWAP / EMAs down


def T(entry=100.0):
    return {"entry": entry, "stop": entry * 0.992, "target": entry * 1.016,
            "opened": datetime.now(timezone.utc), "best": entry}


def test_stop_and_target():
    now = datetime.now(timezone.utc)
    assert "stop" in sc.manage(T(), 99.1, now)
    assert "target" in sc.manage(T(), 101.7, now)
    assert sc.manage(T(), 100.2, now) is None


def test_breakeven_then_lock():
    now, t = datetime.now(timezone.utc), T()
    sc.manage(t, 100.9, now); assert t["stop"] == 100
    sc.manage(t, 101.3, now); assert round(t["stop"], 2) == 100.6
    assert "stop" in sc.manage(t, 100.5, now)   # gave back -> still a +0.6% win


def test_time_stop():
    t = T(); t["opened"] -= timedelta(minutes=25)
    assert "time" in sc.manage(t, 100.1, datetime.now(timezone.utc))


def test_size_is_big_but_capped():
    assert sc.size(100_000, 400_000, 50, 0) == 2000          # 1x equity
    assert sc.size(100_000, 50_000, 50, 1) == 950            # limited by buying power
    assert sc.size(100_000, 400_000, 50, 2) == 0             # max 2 positions


def test_best_trend_picks_strongest_uptrend():
    up = bars([100 + i * 0.05 for i in range(30)])
    up_more = bars([100 + i * 0.1 for i in range(30)])
    down = bars([110 - i * 0.2 for i in range(30)])
    assert sc.best_trend({"A": up, "B": up_more, "C": down}) == "B"
    assert sc.best_trend({"C": down}) is None


def test_no_chasing_extended_moves():
    # strong breakout on volume, but the price ran far above the day's VWAP -> skip
    closes = [100] * 25 + [101, 102, 103, 104, 105] + [106.5]
    vols = [1000] * 30 + [3000]
    assert sc.entry_signal(bars(closes, vols)) == 0


def test_backtest_breakdown_groups_by_symbol_and_time():
    import backtest as bt
    trades = [{"symbol": "TQQQ", "pnl": 100.0, "entry_min": 10},
              {"symbol": "TQQQ", "pnl": -50.0, "entry_min": 40},
              {"symbol": "NVDA", "pnl": -20.0, "entry_min": 45}]
    b = bt.breakdown(trades)
    assert b["by_symbol"]["TQQQ"] == {"n": 2, "pnl": 50.0, "pf": 2.0}
    assert b["by_symbol"]["NVDA"]["pf"] == 0
    assert b["by_entry_time"]["030-059m"]["n"] == 2
    assert bt.breakdown([]) == {"by_symbol": {}, "by_entry_time": {}}


class _O:
    def __init__(self, status, price=None, qty=None, id="x"):
        self.id, self.status, self.filled_avg_price, self.filled_qty = id, status, price, qty


class _T:
    """Fake broker: the order is partly filled first, then fully filled."""
    def __init__(self, seq):
        self.seq = list(seq)

    def get_order_by_id(self, _id):
        return self.seq.pop(0) if len(self.seq) > 1 else self.seq[0]


def _bot(seq, monkeypatch):
    monkeypatch.setattr(sc.time, "sleep", lambda *_: None)
    b = sc.Scalper.__new__(sc.Scalper)
    b.t = _T(seq)
    return b


def test_wait_fill_waits_for_full_fill(monkeypatch):
    from alpaca.trading.enums import OrderStatus
    b = _bot([_O(OrderStatus.PARTIALLY_FILLED, "43.9", "300"), _O(OrderStatus.FILLED, "43.98", "737")], monkeypatch)
    assert b.wait_fill(_O("new"), 737) == (43.98, 737)


def test_wait_fill_gives_up_with_partial(monkeypatch):
    b = _bot([_O("partially_filled", "10.0", "5")], monkeypatch)
    assert b.wait_fill(_O("new"), 9, tries=3) == (10.0, 5)


def test_close_records_real_fill_not_quote(monkeypatch, tmp_path):
    monkeypatch.setattr(sc, "TRADES", str(tmp_path / "t.jsonl"))
    b = _bot([_O("filled", "44.20", "737")], monkeypatch)
    b.dry, b.cooldown = False, {}
    b.trades = {"SQQQ": {"entry": 43.98, "qty": 737}}
    b.cancel_orders = lambda s: None
    b.t.close_position = lambda s: _O("new")
    b.close("SQQQ", "test", 44.23)
    import json
    row = json.loads(open(tmp_path / "t.jsonl").read())
    assert row["exit"] == 44.20 and row["pnl"] == round((44.20 - 43.98) * 737, 2)
