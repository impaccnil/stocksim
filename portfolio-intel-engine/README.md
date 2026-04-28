# AI Portfolio Intelligence Engine (Non-Trading)

This project is a **non-executing** portfolio intelligence system for a retail investor.

## Hard constraints
- **Never executes trades** (no broker integrations, no order routing).
- **No guarantees**: all outputs are probabilistic and uncertainty-aware.
- Focuses on **analysis, simulation, and guidance** only.

## What it does
- Maintains a live portfolio state (`data/portfolio.json`)
- Ingests market/macro/news data through pluggable providers
- Scores each holding using weighted engines:
  - Technical (35%)
  - Fundamental (35%)
  - Macro (20%)
  - Sentiment (10%)
- Detects portfolio risks (concentration, sector clustering, correlation spikes)
- Maintains a shadow **AI simulated portfolio** applying recommendations
- Produces a daily report (email-ready markdown/text)

## Quickstart (local)
1) Install Python 3.11+.
2) Install deps:

```bash
python -m pip install -r requirements.txt
```

3) Run (offline mock data by default):

```bash
python -m portfolio_intel --portfolio data/portfolio.json --out reports
```

The report will be written to `reports/`.

## Email
This repo **renders** the email subject/body but does **not** send email by default.
If you later want SMTP sending, it will be opt-in via environment variables (never hardcoded).

