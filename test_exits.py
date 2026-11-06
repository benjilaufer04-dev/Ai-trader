import exits


def P(entry, price, qty=10):
    return {"qty": qty, "avg_entry": entry, "price": price}


def test_stop_loss_sells_all():
    acts, lv = exits.check({"SPY": P(100, 96)}, {"SPY": {"stop": 97}})
    assert acts[0][:2] == ("SPY", 1.0) and "SPY" not in lv


def test_default_stop_when_none_set():
    acts, lv = exits.check({"GLD": P(100, 97.5)}, {})
    assert not acts and lv["GLD"]["stop"] == 97
    acts, _ = exits.check({"GLD": P(100, 96.9)}, {})
    assert acts and acts[0][1] == 1.0


def test_take_profit_sells_half_once_and_moves_stop_to_entry():
    acts, lv = exits.check({"QQQ": P(100, 106)}, {"QQQ": {"stop": 97, "take_profit": 105}})
    assert acts[0][:2] == ("QQQ", 0.5) and lv["QQQ"]["stop"] == 100
    acts, lv = exits.check({"QQQ": P(100, 107)}, lv)
    assert not acts  # doesn't keep selling every cycle


def test_closed_positions_forgotten():
    _, lv = exits.check({}, {"AAPL": {"stop": 1}})
    assert lv == {}


def test_record_buy_ignores_silly_levels():
    lv = exits.record_buy({}, "SPY", 100, stop=80, take_profit=90)  # stop too far, tp below price
    assert lv["SPY"] == {}
    lv = exits.record_buy({}, "SPY", 100, stop=97, take_profit=106)
    assert lv["SPY"] == {"stop": 97, "take_profit": 106}
