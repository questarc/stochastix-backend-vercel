# Stochastix AI — Features (back end)

FastAPI back end for **Stochastix AI**. It does every calculation; the Angular front end
(separate repository) only displays results. The language model never calculates numbers — it only calls tools.

Results are projections, not guarantees, and not investment advice.

## Feature overview

| Feature | What it does | Endpoints |
|---|---|---|
| 📄 E*TRADE statement reader | Extracts transfer analytics from PDFs, excludes personal identifiers and retains only allowlisted transfer fields in session memory | `GET/DELETE /api/etrade/statement`, `POST /api/etrade/statement/upload` |
| 📄 Fidelity Agent | Fidelity-branded PDF reader with view-only, explicitly labeled portfolio summary values and privacy controls; uses an isolated session store | `GET/DELETE /api/fidelity/statement`, `POST /api/fidelity/statement/upload` |
| 🎲 Simulation | Monte Carlo projection of current and model portfolios; probability of success, percentile bands, histograms | `POST /api/simulations` |
| ⚖️ Rebalance | Drift vs the model portfolio, proposed sells/buys funded by sells, success before/after | `POST /api/rebalance/proposal`, `GET /api/rebalance/settings` |
| 🧾 Orders | Pending order blotter, CSV/JSON export | `POST /api/orders/generate`, `GET/DELETE /api/orders`, `GET /api/orders/export` |
| 📂 Source agents | Sales CSV, A360 Positions (with upload), A360/Orion API placeholders | `/api/sources/...`, `GET /api/accounts/{id}/portfolio` |
| 🤖 Advisor AI | LangChain + LangGraph agent with 15 tools; NVIDIA Nemotron (default), Ollama, OpenAI, Anthropic | `POST /api/chat`, `DELETE /api/chat/{thread_id}`, `GET /api/llm/providers` |
| 🌐 WebLLM support | Exposes the same tools to an agent running in the browser | `GET /api/agent/tools`, `POST /api/agent/tools/{name}` |
| 🔊 VoiceGenAgent | Available in Python development only; Piper TTS is omitted from Docker and Vercel deployments | `POST /api/voice`, `GET /api/voice/{id}` |
| 📰 MarketAnalyzerAgent | Latest economic and market news | `GET /api/market/news` |
| 📈 EarningsAgent | Upcoming earnings reports | `GET /api/market/earnings` |
| 🏦 CorporateActionsAgent | Upcoming dividends and stock splits | `GET /api/market/corporate-actions` |
| 🗺️ MarketHeatmapAgent | Daily % change of the largest US companies by sector (treemap data) | `GET /api/market/heatmap?size=100` |
| 📉 Time Series & Statistical Models | User-uploaded time series analyzed with forward-return linear regression or up-probability logistic regression | `POST /api/time-series/preview`, `POST /api/time-series/analyze` |
| 🔮 Predictive Pricing Agent | PyTorch LSTM trained on recent Yahoo Finance OHLCV data; five daily direction, percent-change and expected-price estimates | `POST /api/predictive-pricing/predict` |
| 📖 Methodology | "How it works" text | `GET /api/methodology` |
| ⚙️ Config | Capital market assumptions and allowed options | `GET /api/config/cma`, `GET /api/config/options` |

Interactive API docs: `http://localhost:8000/docs` (port 8001 when `BACKEND_PORT=8001`).

## Monte Carlo engine (`app/engine/`)
- Sampling: PRNG, Latin Hypercube, scrambled Sobol. Copulas: Gaussian, Student-t (df > 2).
- Seven return distributions (lognormal, normal, uniform, triangular, PERT, exponential, binomial), each matching the
  configured expected return and volatility.
- Yearly loop: contributions before retirement, inflation-adjusted withdrawals after; values floored at zero.
- Same inputs and seed give identical results; seed and settings are logged and returned for audit.
- API returns bands, a 60-bin ending-value histogram and depletion ages — never raw paths (optional `sample_paths` ≤ 100).
  Runs in a thread pool; trials capped at 50,000.

## Rebalancing and orders (`app/rebalance.py`, `app/stores.py`)
- Drift per asset class; rebalance only above `drift_tolerance` (default 2%).
- Overweight holdings sold pro rata; proceeds split across underweight classes by shortfall; whole shares and minimum
  trade value respected; leftover goes to cash.
- Orders: sells first, Market / Day / **Pending**, ID `{account}-{timestamp}-{nnn}`. New orders replace pending orders
  of the same account. Stored in memory behind an `OrderStore` interface.

## Source agents (`app/sources/`)
- **Sales**: accounts, positions and model portfolios from CSV.
- **A360 Positions**: parses A360 exports (`$`, `,`, `(123)`, blank `Qty` → `Custodian Quantity`, CUSIP when no
  ticker, price from `Market Value / Qty`, asset class by ticker override → Sec Type → default) and Fidelity positions
  exports (`Symbol`, `Quantity`, `Last price`, `Current value`). Fidelity rows from all account numbers in the
  uploaded CSV are combined into the selected Quant Models account; account names/numbers and the other performance
  columns are not stored. Money-market rows with blank quantities use current value as quantity at $1/share; pending
  activity and rows without symbols are skipped. Uploads replace an account's positions in memory until reset.
- **A360 API / Orion API**: placeholders (HTTP 501).

## Advisor AI (`app/agent/`)
- `create_agent` + `InMemorySaver` (memory per `thread_id`), recursion limit 20, one agent cached per (provider, model).
- On-screen account/model are prefixed to each message; a client named in the question wins.
- Tools return pre-formatted JSON and catch their own errors:
  `list_accounts`, `list_model_portfolios`, `get_account_portfolio`, `run_simulation` (what-ifs with baseline),
  `compare_current_vs_model`, `propose_rebalance`, `simulate_and_rebalance` (creates pending orders),
  `explain_methodology`, `get_capital_market_assumptions`, `create_audio_summary`,
  `get_market_news`, `get_upcoming_earnings`, `get_corporate_actions`, `get_market_heatmap`.
- Providers (`app/llm.py`): NVIDIA (default `nvidia/nemotron-3.5-lightning-30b-a3b`), Ollama (`qwen2.5:7b`), OpenAI, Anthropic.
  NVIDIA uses the OpenAI-compatible NVIDIA API endpoint. Keys only from environment / `.env`;
  `/api/llm/providers` returns `api_key_set`, never the key.

## VoiceGenAgent (`app/voice/`)
- Rule-based script: strips markdown/tables, rewords figures for speech ("76.8%" → "76.8 percent",
  "$1,380,575" → "1,380,575 dollars") **without changing values**, adds opening and advisor name, and always ends with
  the fixed disclaimer.
- Piper TTS (`en_US-lessac-medium`, free, offline; text never leaves the server). Available in local development through `requirements-dev.txt`; excluded from Docker and Vercel.
- Newest 50 clips kept in memory; file name `{account}_summary_{timestamp}.wav`.

## Market agents (`app/market/`, `config/market.json`)
- Free sources, **no API key**: Federal Reserve (press, speeches), BEA, SEC, ECB, MarketWatch RSS; Nasdaq public
  earnings, dividends and splits calendars.
- News tagged by topic, de-duplicated, sorted newest first; cached 15 minutes. Calendars cached 6 hours.
- Earnings filtered to client holdings, the configurable watchlist and companies ≥ $50B market cap.
- Heatmap: Nasdaq public screener; top N companies by market cap (default 100, 10–300, duplicate share classes
  excluded), grouped by sector with market-cap-weighted sector and market change and up/down breadth; cached 15 minutes.
- Items held by clients carry `held_by` account IDs. Failed sources are reported per source; nothing is invented.
- Nasdaq and MarketWatch feeds are public but unofficial; check terms or use a licensed vendor for production.

## Time Series & Statistical Models (`app/time_series.py`)
- CSV uploads (20 MB / 100,000 rows) are analyzed in memory; preview returns column names and type metadata only.
- Pick a date column, positive asset-price target, numeric predictors, and a 1–30 observation horizon.
- Linear model target is the forward simple return `(price[t+h] / price[t]) - 1`; logistic target is whether that return
  is positive. Predictors use values available at date `t`, and dates are sorted chronologically.
- Incomplete observations are removed; predictors are standardized using the training portion only. The first 80% of
  complete observations train the model, with the final 20% reserved as a chronological holdout.
- Linear output includes test R²/MAE/RMSE and coefficient standard errors/p-values; logistic output includes test
  accuracy, precision, recall, ROC AUC, log loss, confusion counts, and odds ratios. No uploaded observations are
  retained. Associations are not causal estimates, and a holdout is not a validated trading strategy.

## Predictive Pricing Agent (`app/predictive_pricing.py`)
- `POST /api/predictive-pricing/predict` accepts a stock ticker and a positive user-entered current price.
- Downloads daily Open, High, Low, Close and Volume values from Yahoo Finance for the latest 30 calendar days. At least 12 complete observations are required; the model is trained for each request and no history or model state is retained.
- A CPU PyTorch LSTM recursively estimates the next five daily percentage changes and directional labels (`up` / `down`). The response includes expected prices, forecast dates, recent closes and the observation count.
- The entered price anchors the expected-price sequence; historical closes are rescaled to meet that anchor on the chart. The model uses a short sample and its estimates are experimental, not investment advice or validated trading signals.
- PyTorch adds a large runtime dependency. Verify the hosting provider's serverless function bundle limits before deploying; the Docker backend is the supported path if the PyTorch runtime does not fit the Vercel function limit.

## E*TRADE PDF reader (`app/etrade.py`)
- Upload a text-based PDF up to 20 MB and 500 pages; extracted text is limited to 2 million characters.
- The PDF and unredacted extracted text are processed transiently in memory and never written to disk or a database.
- The response preview is generated from allowlisted transfer dates, statuses, and amounts; the original filename,
  full statement text, account labels, names, SSNs, passwords, and addresses are not retained.
- Only allowlisted analytics, or a sanitized structure preview for an unrecognized statement, are retained in process
  memory per browser session until cleared or backend restart.
- The session key is not authentication; do not expose financial statements through an untrusted deployment.
- Transfer activity parsing reports transaction count, listed amount, completed amount and scheduled amount. It does
  not infer a portfolio balance, holdings or investment performance. Analytics are view-only; no brokerage actions
  are submitted.
- Defense-in-depth text redaction recognizes common labeled identity, SSN, credential and address fields, but the
  displayed preview is allowlist-generated rather than relying on pattern redaction to identify every personal value.
- Scanned/image-only and password-protected PDFs are not supported. Transfer activity is parsed as listed cash flows.
- Fidelity additionally extracts the overview's portfolio value, change from last period, and total including other
  holdings when the three labels precede exactly three decimal amounts in the first-page summary block. From its
  Portfolio Summary section it also extracts total income for this period/year-to-date and aggregate Top Holdings
  value/portfolio percentage. Labeled per-account balance, activity, income, and aggregate Top Holdings metrics are
  also retained without account names/numbers or individual holding descriptions. Holdings rows retain only their
  numeric values (beginning/ending market value, quantity, unit price, cost basis, unrealized gain/loss, and EAI/EY).
  The Activity section displays settlement date, security, description, quantity, price, cost basis, transaction
  cost, and amount from recognized transaction tables; requested security fields are retained with those rows.
  Fidelity Agent also accepts positions CSV exports and retains only the displayed position fields, excluding account
  names and numbers.
  It does not infer investment returns; other layouts receive a sanitized structure preview.
  Unrecognized Fidelity previews may include safe parser diagnostics (label count, boundary presence, and amount count),
  never extracted financial values.
  Fidelity uses separate endpoints and an isolated in-memory session store.

## Configuration
| File / variable | Purpose |
|---|---|
| `config/capital_market_assumptions.json` | Returns, volatility, correlation, distributions, inflation |
| `config/rebalance.json` | Drift tolerance, minimum trade, whole shares, default securities |
| `config/sources.json` | Enabled source agents and file paths |
| `config/llm.json` | Providers, models, temperature, timeout, max trials per tool call |
| `config/market.json` | News feeds, calendar URLs, watchlist, cache times |
| `requirements.txt` | Includes PyTorch for the LSTM agent and yfinance for daily market history |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | LLM keys (`.env`, git- and docker-ignored) |
| `OLLAMA_BASE_URL`, `MCAPP_LLM_PROVIDER`, `MCAPP_LLM_MODEL` | LLM overrides |
| `MCAPP_VOICE`, `MCAPP_VOICES_DIR` | Piper voice |
| `CORS_ORIGINS`, `MCAPP_LOG_LEVEL` | CORS and logging |

## Run
```powershell
python -m venv .venv; .venv\Scripts\activate
pip install -r requirements.txt
python -m piper.download_voices --download-dir voices en_US-lessac-medium
uvicorn app.main:app --reload --port 8000
pytest            # 98 tests
```
Docker: `Dockerfile` (python:3.12-slim, health check `GET /api/health`). With the front end:
`docker compose up -d --build` from the parent folder (app on `http://localhost:8080`).

## Known limitations
- Orders, uploads, imported E*TRADE statement text, chat memory and audio are in memory and lost on restart (stores sit behind interfaces so a
  database can replace them).
- No login; anyone who can reach the app can use every feature.
- Audio sent to investors is a client communication; follow your firm's review and record-keeping rules.
