"""The trader's memory: a decision diary, its track record, and lessons from daily reviews.

Files live in memory/ inside the repo, and the GitHub workflows commit them back after
every run, so the bot remembers across runs.
  memory/decisions_stocks.jsonl / decisions_crypto.jsonl   one line per decision
  memory/lessons.md                                         rules the AI wrote for itself
  memory/reviews/YYYY-MM-DD.md                              each daily review
"""
import json
import os
from datetime import datetime, timedelta, timezone

DIR = os.getenv("MEMORY_DIR", "memory")


def _path(crypto):
    return os.path.join(DIR, f"decisions_{'crypto' if crypto else 'stocks'}.jsonl")


def log_decisions(crypto, market_view, decisions, statuses, snapshots):
    """Append this cycle's decisions with the price at decision time."""
    os.makedirs(DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat(timespec="minutes")
    with open(_path(crypto), "a") as f:
        if not decisions:
            f.write(json.dumps({"ts": ts, "action": "hold_all", "market_view": market_view[:400]}) + "\n")
        for d in decisions:
            sym = str(d.get("symbol", "")).upper()
            f.write(json.dumps({
                "ts": ts, "symbol": sym, "action": d.get("action"),
                "confidence": d.get("confidence"), "usd": d.get("buy_usd"),
                "sell_fraction": d.get("sell_fraction"),
                "price": (snapshots.get(sym) or {}).get("price"),
                "status": statuses.get(sym, "unknown"),
                "reasoning": str(d.get("reasoning", ""))[:400],
                "exit_plan": str(d.get("exit_plan", ""))[:200],
                "market_view": market_view[:300],
            }) + "\n")


def load_decisions(crypto=None, days=14):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    out = []
    for c in ([crypto] if crypto is not None else [False, True]):
        p = _path(c)
        if not os.path.exists(p):
            continue
        for line in open(p):
            try:
                r = json.loads(line)
                if datetime.fromisoformat(r["ts"]) >= cutoff:
                    r["mode"] = "crypto" if c else "stocks"
                    out.append(r)
            except Exception:
                pass
    return out


def outcomes(decisions, prices, benchmark=None):
    """For each executed buy/sell: % move of the symbol since the decision
    (for a sell, a falling price afterwards means the sell was right)."""
    rows = []
    bench_now = prices.get(benchmark) if benchmark else None
    for r in decisions:
        if r.get("status") != "executed" or not r.get("price") or r.get("symbol") not in prices:
            continue
        move = (prices[r["symbol"]] / r["price"] - 1) * 100
        rows.append({
            "ts": r["ts"][:16], "symbol": r["symbol"], "action": r["action"],
            "confidence": r.get("confidence"), "price_then": r["price"], "price_now": prices[r["symbol"]],
            "move_since_%": round(move, 2),
            "result": ("good" if (move > 0) == (r["action"] == "buy") else "bad") if abs(move) > 0.3 else "flat",
            "reasoning": r.get("reasoning", "")[:160],
        })
    return rows


def track_record(rows):
    """Compact summary the trader sees before each decision."""
    if not rows:
        return {"note": "No completed trades to learn from yet."}
    buys = [r for r in rows if r["action"] == "buy"]
    good = sum(r["result"] == "good" for r in rows)
    bad = sum(r["result"] == "bad" for r in rows)
    per_symbol = {}
    for r in buys:
        per_symbol.setdefault(r["symbol"], []).append(r["move_since_%"])
    return {
        "decisions_scored": len(rows), "good": good, "bad": bad, "flat": len(rows) - good - bad,
        "avg_move_after_buy_%": round(sum(r["move_since_%"] for r in buys) / len(buys), 2) if buys else None,
        "avg_move_after_buy_by_symbol_%": {k: round(sum(v) / len(v), 2) for k, v in per_symbol.items()},
        "last_10": rows[-10:],
    }


def load_lessons():
    p = os.path.join(DIR, "lessons.md")
    if not os.path.exists(p):
        return ["No lessons yet - this is the first period of trading."]
    return [l[2:].strip() for l in open(p) if l.startswith("- ")][:12]


REVIEW_SYSTEM = """You are the head of a small trading desk reviewing your own AI trader's recent
decisions. Your job is to make it a better trader by writing a short list of LESSONS that it
will read before every future decision.

You get: the account and positions now, every recent decision with the price at the time and
how the price moved afterwards, how the benchmark moved, and the current lessons list.

Rules for lessons:
- Base every lesson on evidence in the data (cite the trades/numbers briefly). Do not invent.
- Prefer lessons that held across several trades. One trade is noise: phrase single-trade
  observations cautiously ("watch for...") or leave them out.
- Make them concrete and actionable (entry timing, sizing, when to exit, which signals
  worked or failed, over-trading, chasing, concentration, fees on crypto).
- Keep lessons that still look right, rewrite ones the new data contradicts, drop stale ones.
- Maximum 12 lessons, one sentence each, most important first.
- Never suggest breaking or loosening the hard risk limits, using margin, or shorting.
- Be honest: if it is too early to conclude much, say so in the summary and keep lessons modest."""

REVIEW_TOOL = {
    "name": "submit_review",
    "description": "Daily performance review and updated lessons.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "4-8 sentences: how the trader did and why."},
            "what_worked": {"type": "array", "items": {"type": "string"}},
            "mistakes": {"type": "array", "items": {"type": "string"}},
            "lessons": {"type": "array", "items": {"type": "string"}, "description": "Full updated list, max 12."},
        },
        "required": ["summary", "mistakes", "lessons"],
    },
}


def save_review(review, account_line):
    os.makedirs(os.path.join(DIR, "reviews"), exist_ok=True)
    day = datetime.now(timezone.utc).date().isoformat()
    lessons = [str(l).strip() for l in review.get("lessons", []) if str(l).strip()][:12]
    with open(os.path.join(DIR, "lessons.md"), "w") as f:
        f.write(f"# Lessons the trader has learned (updated {day})\n\n")
        f.writelines(f"- {l}\n" for l in lessons)
    with open(os.path.join(DIR, "reviews", f"{day}.md"), "w") as f:
        f.write(f"# Daily review {day}\n\n{account_line}\n\n## Summary\n{review.get('summary', '')}\n\n")
        f.write("## What worked\n" + "".join(f"- {x}\n" for x in review.get("what_worked", [])) + "\n")
        f.write("## Mistakes\n" + "".join(f"- {x}\n" for x in review.get("mistakes", [])) + "\n")
        f.write("## Lessons\n" + "".join(f"- {x}\n" for x in lessons))
    return lessons
