"""SQLite log of every cycle, decision and order, so you can audit what the AI did and why."""
import json
import sqlite3
from datetime import datetime, timezone


class Journal:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS cycles(id INTEGER PRIMARY KEY, ts TEXT, equity REAL,
            cash REAL, market_view TEXT, raw TEXT);
        CREATE TABLE IF NOT EXISTS decisions(id INTEGER PRIMARY KEY, cycle_id INT, ts TEXT,
            symbol TEXT, action TEXT, confidence REAL, status TEXT, detail TEXT, reasoning TEXT);
        CREATE TABLE IF NOT EXISTS orders(id INTEGER PRIMARY KEY, ts TEXT, symbol TEXT, side TEXT,
            notional REAL, qty REAL, order_id TEXT, status TEXT);
        """)

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def cycle(self, account, result):
        cur = self.db.execute(
            "INSERT INTO cycles(ts,equity,cash,market_view,raw) VALUES(?,?,?,?,?)",
            (self.now(), account["equity"], account["cash"],
             result.get("market_view", ""), json.dumps(result)))
        self.db.commit()
        return cur.lastrowid

    def decision(self, cycle_id, d, status, detail=""):
        self.db.execute(
            "INSERT INTO decisions(cycle_id,ts,symbol,action,confidence,status,detail,reasoning)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (cycle_id, self.now(), d.get("symbol"), d.get("action"), d.get("confidence"),
             status, detail, d.get("reasoning", "")))
        self.db.commit()

    def order(self, a, order_id, status):
        self.db.execute(
            "INSERT INTO orders(ts,symbol,side,notional,qty,order_id,status) VALUES(?,?,?,?,?,?,?)",
            (self.now(), a.symbol, a.side, a.notional, a.qty, order_id, status))
        self.db.commit()

    def trades_today(self):
        today = datetime.now(timezone.utc).date().isoformat()
        return self.db.execute(
            "SELECT COUNT(*) FROM orders WHERE ts LIKE ? AND status != 'error'",
            (today + "%",)).fetchone()[0]

    def recent(self, n=15):
        rows = self.db.execute(
            "SELECT ts,symbol,action,confidence,status,reasoning FROM decisions "
            "ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [dict(zip(["ts", "symbol", "action", "confidence", "status", "reasoning"], r))
                for r in rows]
