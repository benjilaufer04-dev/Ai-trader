#!/bin/bash
cd "$(dirname "$0")"
echo ""
echo "=== AI TRADER ==="
if [ ! -f .env ]; then
  echo "Prima configurazione: incolla le chiavi quando te le chiedo (Cmd+V poi Invio)."
  echo ""
  read -r -p "1/3  Incolla la ALPACA KEY (inizia con PK): " AK
  read -r -p "2/3  Incolla l'ALPACA SECRET: " AS
  read -r -p "3/3  Incolla la GEMINI KEY (inizia con AQ.): " GK
  sed -e "s|^ALPACA_API_KEY=.*|ALPACA_API_KEY=${AK// /}|" \
      -e "s|^ALPACA_SECRET_KEY=.*|ALPACA_SECRET_KEY=${AS// /}|" \
      -e "s|^GEMINI_API_KEY=.*|GEMINI_API_KEY=${GK// /}|" .env.example > .env
  echo "Chiavi salvate."
fi
if [ ! -d .venv ]; then
  echo "Installo (solo la prima volta, 1-2 minuti)..."
  python3 -m venv .venv || { echo "Manca Python: se il Mac chiede di installare gli strumenti, clicca Installa e poi rilancia."; exit 1; }
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt || { echo "Installazione fallita, manda uno screenshot a Claude."; exit 1; }
fi
echo ""
echo "AI trader ACCESO (soldi finti). Per spegnerlo chiudi questa finestra."
echo ""
.venv/bin/python agent.py
