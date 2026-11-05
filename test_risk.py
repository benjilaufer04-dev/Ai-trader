"""Run: python -m pytest -q   (no API keys needed)"""
from config import Settings
import risk

S = Settings()
S.watchlist = ["AAPL", "MSFT", "NVDA", "SPY"]
S.kill_switch_file = "__no_such_file__"
ACCT = {"equity": 100_000, "last_equity": 100_000, "cash": 50_000, "buying_power": 50_000}


def buy(sym, usd, conf=0.8):
    return {"symbol": sym, "action": "buy", "buy_usd": usd, "confidence": conf, "reasoning": "t"}


def test_buy_capped_by_max_order():
    ok, _ = risk.check([buy("AAPL", 50_000)], ACCT, {}, 0, S)
    assert ok[0].notional == S.max_order_usd


def test_position_cap():
    pos = {"AAPL": {"qty": 10, "market_value": 9_500}}
    ok, _ = risk.check([buy("AAPL", 2_000)], ACCT, pos, 0, S)
    assert ok[0].notional == 500  # 10% of 100k = 10k, already hold 9.5k


def test_rejects_offlist_lowconf_short():
    ds = [buy("GME", 1000), buy("MSFT", 1000, conf=0.3),
          {"symbol": "NVDA", "action": "sell", "sell_fraction": 1, "confidence": 0.9}]
    ok, rej = risk.check(ds, ACCT, {}, 0, S)
    assert not ok and len(rej) == 3


def test_daily_loss_blocks_buys_not_sells():
    acct = dict(ACCT, equity=96_000)
    pos = {"MSFT": {"qty": 5, "market_value": 2_000}}
    ds = [buy("AAPL", 1000), {"symbol": "MSFT", "action": "sell", "sell_fraction": 0.5, "confidence": 0.9}]
    ok, rej = risk.check(ds, acct, pos, 0, S)
    assert [a.side for a in ok] == ["sell"] and ok[0].qty == 2.5
    assert "loss limit" in rej[0][1]


def test_cash_reserve_and_trade_limit():
    acct = dict(ACCT, cash=20_500)  # reserve is 20k => only $500 room
    ok, _ = risk.check([buy("AAPL", 1000)], acct, {}, 0, S)
    assert ok[0].notional == 500
    ok, rej = risk.check([buy("SPY", 1000)], ACCT, {}, S.max_trades_per_day, S)
    assert not ok and "trade limit" in rej[0][1]


def test_kill_switch(tmp_path):
    f = tmp_path / "STOP"; f.write_text("")
    S2 = Settings(); S2.watchlist = S.watchlist; S2.kill_switch_file = str(f)
    ok, rej = risk.check([buy("AAPL", 1000)], ACCT, {}, 0, S2)
    assert not ok and "kill switch" in rej[0][1]


def test_market_closed():
    ok, rej = risk.check([buy("AAPL", 1000)], ACCT, {}, 0, S, market_open=False)
    assert not ok


def test_can_sell_holding_not_on_watchlist():
    pos = {"TSLA": {"qty": 4, "market_value": 1000}}
    ds = [{"symbol": "TSLA", "action": "sell", "sell_fraction": 1, "confidence": 0.9},
          buy("TSLA", 500)]
    ok, rej = risk.check(ds, ACCT, pos, 0, S)
    assert [a.side for a in ok] == ["sell"] and "watchlist" in rej[0][1]
