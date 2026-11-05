"""The AI reads the market snapshot and returns structured trade decisions.

Providers: "gemini" (free tier, default) or "anthropic" (Claude, paid API)."""
import json


SYSTEM = """You are a disciplined portfolio manager running a small account (US stocks/ETFs or crypto - see asset_class).
Each cycle you get: account state, current positions, technical snapshots for a fixed
watchlist, recent news, and your own recent trades. Decide what to do NOW.

Principles:
- Doing nothing is usually the right answer. Only act on a clear, specific edge.
- Prefer trend-following entries with confirmation (price above rising averages, healthy
  RSI, supportive news) over catching falling knives.
- Cut losers that have broken their thesis; let winners run but trim if extended.
- Respect concentration: diversify across sectors, don't pile into correlated names.
- Never chase a single headline. Never trade a symbol outside the watchlist.
- Be honest with confidence (0-1). Below 0.65 will be ignored, so use that deliberately.
- You can only go long. "sell" means reduce or close an existing position.

Hard risk limits are enforced by separate code after you; you don't need to restate them,
but proposals that break them are simply rejected.

Always respond with your decisions in the required format, nothing else."""

TOOL = {
    "name": "submit_decisions",
    "description": "Submit this cycle's market view and trade decisions.",
    "input_schema": {
        "type": "object",
        "properties": {
            "market_view": {
                "type": "string",
                "description": "2-4 sentences on the overall market regime right now.",
            },
            "decisions": {
                "type": "array",
                "description": "Only include symbols you want to buy or sell. Empty list = hold everything.",
                "items": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "action": {"type": "string", "enum": ["buy", "sell"]},
                        "buy_usd": {"type": "number", "description": "Dollar amount to buy (buy only)."},
                        "sell_fraction": {
                            "type": "number",
                            "description": "Fraction of current position to sell, 0-1 (sell only). 1 = close.",
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "reasoning": {"type": "string"},
                        "exit_plan": {"type": "string", "description": "When you'd get out (buys)."},
                    },
                    "required": ["symbol", "action", "confidence", "reasoning"],
                },
            },
        },
        "required": ["market_view", "decisions"],
    },
}


GEMINI_FALLBACKS = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-2.5-flash-lite"]


class Brain:
    def __init__(self, provider: str, api_key: str, model: str):
        self.provider, self.model = provider, model
        if provider == "gemini":
            from google import genai
            self.client = genai.Client(api_key=api_key)
        else:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)

    def decide(self, context: dict) -> dict:
        prompt = "Current state (JSON):\n" + json.dumps(context, indent=1, default=str)
        return self._gemini(prompt) if self.provider == "gemini" else self._claude(prompt)

    def _gemini(self, prompt):
        import time
        from google.genai import types
        cfg = types.GenerateContentConfig(
            system_instruction=SYSTEM,
            temperature=0.3,
            response_mime_type="application/json",
            response_json_schema=TOOL["input_schema"],
        )
        models = [self.model] + [m for m in GEMINI_FALLBACKS if m != self.model]
        last_err = None
        # Free-tier models are sometimes overloaded (503) or rate-limited (429):
        # retry each model a few times with a pause, then move on to the next model.
        for m in models:
            for attempt in range(3):
                try:
                    r = self.client.models.generate_content(model=m, contents=prompt, config=cfg)
                    if m != self.model:
                        print(f"(used backup model {m})")
                    return json.loads(r.text)
                except Exception as e:
                    last_err = e
                    msg = str(e).lower()
                    if "404" in msg or "not found" in msg:
                        break  # model doesn't exist -> next model
                    if any(x in msg for x in ("503", "unavailable", "overloaded", "429",
                                              "resource_exhausted", "500", "internal", "deadline")):
                        time.sleep(10 * (attempt + 1))
                        continue
                    raise  # anything else (bad key etc.) -> show the real error
        raise last_err

    def _claude(self, prompt):
        msg = self.client.messages.create(
            model=self.model, max_tokens=4000, system=SYSTEM,
            tools=[TOOL], tool_choice={"type": "tool", "name": "submit_decisions"},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in msg.content:
            if block.type == "tool_use":
                return block.input
        return {"market_view": "no tool call returned", "decisions": []}
