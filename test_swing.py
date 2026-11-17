"""Tests for the swing bot (IBS) and its coexistence with the scalper. No network."""
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pandas as pd

import scalper as sc
import swing as sw


def test_ibs_math():
    assert sw.ibs(110, 100, 100) == 0 and sw.ibs(110, 100, 110) == 1 and sw.ibs(100, 100, 100) == 0.5


def test_decide_buy_hold_sell():
    assert sw.decide(False, 0.1, 0) == "buy"
    assert sw.decide(False, 0.5, 0) == "wait"
    assert sw.decide(True, 0.1, 1) == "hold"           # closed weak again -> one more day
    assert sw.decide(True, 0.6, 1) == "sell"
    assert sw.decide(True, 0.1, sw.CFG["max_days"]) == "sell"   # never hold forever


def test_size_small_and_never_leveraged():
    assert sw.qty_for(100_000, 100_000, 300) == 33          # 10% of the account
    assert sw.qty_for(100_000, 1_000, 300) == 3             # capped by cash, no margin
    assert sw.qty_for(100_000, -5_000, 300) == 0


def test_timing_plan():
    assert sw.minutes_plan(95) == "skip"                    # the other (DST) scheduled run handles today
    assert sw.minutes_plan(1) == "late"
    assert sw.minutes_plan(35) == 30
    assert sw.minutes_plan(4) == 0


class FakeTrading:
    def __init__(self, positions=(), fill="300.0"):
        self.positions, self.orders, self.fill, self.cancelled = list(positions), [], fill, 0

    def get_clock(self):
        now = datetime.now(timezone.utc)
        return NS(is_open=True, timestamp=now, next_close=now + timedelta(minutes=5))

    def get_account(self):
        return NS(equity="100000", cash="90000", last_equity="100000", buying_power="200000")

    def get_all_positions(self):
        return self.positions

    def submit_order(self, req):
        self.orders.append(req)
        return NS(id=f"o{len(self.orders)}")

    def get_order_by_id(self, oid):
        n = int(self.orders[int(oid[1:]) - 1].qty)
        return NS(status="filled", filled_qty=str(n), filled_avg_price=self.fill)

    def get_orders(self, req):
        return []

    def cancel_order_by_id(self, oid):
        self.cancelled += 1


def _swing(monkeypatch, tmp_path, trading, last_price, state=None):
    monkeypatch.setattr(sw, "TRADES", str(tmp_path / "s.jsonl"))
    monkeypatch.setattr(sw, "STATE", str(tmp_path / "state.json"))
    if state:
        (tmp_path / "state.json").write_text(json.dumps(state))
    monkeypatch.setattr(sw.time, "sleep", lambda *_: None)
    b = sw.Swing.__new__(sw.Swing)
    b.t, b.dry = trading, False
    b.state = state or {}
    idx = pd.date_range("2026-11-16 14:30", periods=380, freq="1min", tz="UTC")
    bars = pd.DataFrame({"high": 310.0, "low": 290.0, "close": 300.0}, index=idx)
    b.today_bars = lambda sym, start: bars
    b.last_price = lambda sym: last_price
    return b


def test_buys_weak_close_with_broker_stop(monkeypatch, tmp_path):
    t = FakeTrading()
    b = _swing(monkeypatch, tmp_path, t, last_price=291.0)     # IBS 0.05
    b.run(now_mode=True)
    buy, stop = t.orders
    assert buy.side.value == "buy" and int(buy.qty) == 34          # $10k / $291
    assert stop.side.value == "sell" and abs(float(stop.stop_price) - 282.0) < 0.01
    assert json.loads((tmp_path / "s.jsonl").read_text().splitlines()[0])["side"] == "buy"


def test_sells_after_bounce(monkeypatch, tmp_path):
    pos = NS(symbol="SMH", qty="33", avg_entry_price="291.0")
    t = FakeTrading([pos], fill="305.0")
    st = {"SMH": {"entry": 291.0, "qty": 33, "days": 0, "last_day": "2000-01-01"}}
    b = _swing(monkeypatch, tmp_path, t, last_price=305.0, state=st)   # IBS 0.75
    b.run(now_mode=True)
    assert t.orders[0].side.value == "sell"
    row = json.loads((tmp_path / "s.jsonl").read_text().splitlines()[-1])
    assert row["side"] == "sell" and row["pnl"] == round((305 - 291) * 33, 2)
    assert json.loads((tmp_path / "state.json").read_text()) == {}


def test_refuses_live_money(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("PAPER", "false")
    try:
        sw.Swing()
        assert False, "must refuse"
    except SystemExit as e:
        assert "paper" in str(e).lower()


def test_scalper_leaves_swing_position_alone(monkeypatch):
    closed = []
    b = sc.Scalper.__new__(sc.Scalper)
    b.t = NS(get_all_positions=lambda: [NS(symbol="SMH", qty="33", avg_entry_price="300", current_price="301"),
                                        NS(symbol="GLD", qty="5", avg_entry_price="200", current_price="201")])
    b.close = lambda sym, why, px: closed.append(sym)
    b.trades = {}
    b.adopt_positions()
    assert closed == ["GLD"]
