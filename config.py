"""All settings in one place. Values come from .env (see .env.example)."""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _f(name, default):
    return float(os.getenv(name, default))


def _i(name, default):
    return int(os.getenv(name, default))


@dataclass
class Settings:
    # --- keys ---
    alpaca_key: str = os.getenv("ALPACA_API_KEY", "")
    alpaca_secret: str = os.getenv("ALPACA_SECRET_KEY", "")
    # AI provider: "gemini" (free tier) or "anthropic" (Claude, paid)
    ai_provider: str = os.getenv("AI_PROVIDER", "gemini").lower()
    gemini_key: str = os.getenv("GEMINI_API_KEY", "")
    anthropic_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    model: str = os.getenv("AI_MODEL", "")

    @property
    def ai_key(self):
        return self.gemini_key if self.ai_provider == "gemini" else self.anthropic_key

    @property
    def ai_model(self):
        return self.model or ("gemini-flash-latest" if self.ai_provider == "gemini"
                              else "claude-sonnet-5-5")

    # --- paper vs live ---
    # Live trading requires BOTH paper=false AND the exact confirmation phrase.
    paper: bool = os.getenv("PAPER", "true").lower() != "false"
    live_confirm: str = os.getenv("LIVE_TRADING_CONFIRM", "")

    # --- what it may trade ---
    watchlist: list = field(default_factory=lambda: [
        s.strip().upper() for s in os.getenv(
            "WATCHLIST",
            "SPY,QQQ,AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,AMD,JPM,XLE,XLV,TLT,GLD",
        ).split(",") if s.strip()
    ])

    # --- schedule ---
    interval_minutes: int = _i("INTERVAL_MINUTES", 60)

    # --- risk limits (the AI can never override these) ---
    max_position_pct: float = _f("MAX_POSITION_PCT", 0.10)      # max 10% of equity in one stock
    max_order_usd: float = _f("MAX_ORDER_USD", 2000)            # max $ per single buy
    max_trades_per_day: int = _i("MAX_TRADES_PER_DAY", 6)
    daily_loss_limit_pct: float = _f("DAILY_LOSS_LIMIT_PCT", 0.03)  # -3% on the day => no new buys
    min_cash_reserve_pct: float = _f("MIN_CASH_RESERVE_PCT", 0.20)  # always keep 20% in cash
    min_confidence: float = _f("MIN_CONFIDENCE", 0.65)
    max_positions: int = _i("MAX_POSITIONS", 8)

    kill_switch_file: str = os.getenv("KILL_SWITCH_FILE", "STOP")
    db_path: str = os.getenv("DB_PATH", "journal.db")

    LIVE_PHRASE = "I accept that real money can be lost"

    def validate(self):
        missing = [n for n, v in [("ALPACA_API_KEY", self.alpaca_key),
                                  ("ALPACA_SECRET_KEY", self.alpaca_secret),
                                  ("GEMINI_API_KEY" if self.ai_provider == "gemini"
                                   else "ANTHROPIC_API_KEY", self.ai_key)] if not v]
        if missing:
            raise SystemExit(f"Missing in .env: {', '.join(missing)}")
        if not self.paper and self.live_confirm != self.LIVE_PHRASE:
            raise SystemExit(
                "PAPER=false but LIVE_TRADING_CONFIRM is not set to the exact phrase:\n"
                f'  LIVE_TRADING_CONFIRM="{self.LIVE_PHRASE}"\nRefusing to trade real money.'
            )


settings = Settings()
