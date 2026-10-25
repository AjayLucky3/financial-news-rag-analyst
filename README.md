# Financial News Sentiment + RAG Analyst

A production-grade, modular financial news analytics and evidence-grounded Retrieval-Augmented Generation (RAG) system built in Python.

This project ingests financial headlines for Indian market equities via **GDELT DOC 2.0 API** (with **Google News RSS** fallback), performs sentiment scoring using **ProsusAI FinBERT**, aligns news events with actual **yfinance** market trading sessions, stores structured data in **SQLite** and vector embeddings in **ChromaDB**, conducts non-causal statistical correlation and shock analysis, and provides a deterministic RAG query engine integrated into an executive **Streamlit** dashboard.

> [!IMPORTANT]
> **Methodological Safety Notice**: This system is designed strictly as an empirical analytics and quantitative research tool. It measures observed statistical associations and correlations. It does **NOT** claim causation, does **NOT** provide investment advice (no buy/sell/hold recommendations), and is **NOT** a stock price predictor.

---

## 🏗️ Architecture & Component Responsibilities

```
GDELT DOC 2.0 API / Google News RSS Fallback
                     │
                     ▼
           News Cleaning & Deduplication
                     │
                     ▼
           FinBERT Sentiment Scoring
           (ProsusAI/finbert PyTorch)
                     │
                     ▼
        Trading-Session Alignment & Returns
             (yfinance Trading Dates)
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
  SQLite Database           ChromaDB Vector Store
 (data/finance.db)            (data/chroma_db)
 Numerical Source of Truth   Textual Evidence Layer
        │                         │
        └────────────┬────────────┘
                     ▼
         Deterministic RAG Router
      (TEMPORAL / AGGREGATE / SEMANTIC)
                     │
                     ▼
       Grounded Evidence Context & Citations
                     │
                     ▼
         External LLM Synthesis
      (Groq API / Hugging Face Fallback)
                     │
                     ▼
      Streamlit Executive Dashboard (app.py)
```

### Component Responsibilities
- **GDELT DOC 2.0 API**: Primary news retrieval source with `YYYYMMDDHHMMSS` parameters, validated HTTP parsing, and exponential backoff retry.
- **Google News RSS**: Automated fallback news retriever when GDELT is empty or rate-limited.
- **yfinance**: Sequential price retriever defining valid market trading session dates.
- **ProsusAI FinBERT**: Primary sentiment model (`model.eval()` mode, `torch.no_grad()` inference in bounded memory batches).
- **SQLite (`data/finance.db`)**: Structured numerical source of truth (`headlines` and `daily_metrics` tables). Missing news days remain strictly `NULL`/`NaN` (never zero-imputed).
- **ChromaDB (`data/chroma_db`)**: Textual vector retrieval layer using `sentence-transformers/all-MiniLM-L6-v2` embeddings with scalar metadata filtering.
- **Statistical Analysis**: Calculates Pearson ($r$), Spearman ($\rho$), 7-observation and 14-observation rolling correlations, $2.0\sigma$ sentiment shock detection, two-session forward returns, and GDELT/FinBERT signal cross-validation.
- **Deterministic RAG Router**: Pure Python query classification routing questions to `TEMPORAL`, `AGGREGATE`, or `SEMANTIC` pipelines without exposing dynamic SQL to the LLM.
- **External LLM (Groq / HuggingFace)**: Natural-language synthesis **ONLY** grounded in retrieved evidence. Never computes statistics, invents URLs/facts, or executes commands.
- **Streamlit Dashboard (`app.py`)**: 3-tab interactive executive UI displaying metrics, shock events, and the AI Analyst interface.

---

## 🛠️ Technology Stack

- **Language**: Python 3.10+
- **Data Manipulation**: Pandas, NumPy, SciPy
- **Machine Learning & NLP**: PyTorch, Hugging Face Transformers (`ProsusAI/finbert`), Sentence Transformers (`all-MiniLM-L6-v2`)
- **Storage**: SQLite3, ChromaDB
- **Financial & News APIs**: `yfinance`, GDELT DOC 2.0 API, `feedparser`, BeautifulSoup4
- **LLM Integrations**: Groq Cloud API (`llama-3.1-8b-instant`), Hugging Face Inference API
- **User Interface & Visualizations**: Streamlit, Plotly
- **Testing**: Pytest, Tenacity

---

## 🚀 Key Features

1. **Robust Dual News Fetcher**: Queries GDELT DOC 2.0 API with exponential backoff retries (HTTP 429/5xx handling) and seamlessly falls back to Google News RSS.
2. **FinBERT Bounded Sentiment Analysis**: High-precision financial sentiment scoring ($[-1.0, +1.0]$) using model evaluation mode and bounded batches under a 12 GB RAM budget.
3. **Market Session Alignment**: Maps weekend and holiday headlines to the next actual market trading session in `yfinance` price history.
4. **Strict `NaN` Preservation**: Missing news days remain `NaN`/`NULL` for sentiment metrics rather than being falsely imputed with zero.
5. **Non-Naive Return Calculation**: Computes next-day returns based on consecutive trading dates, omitting market closures.
6. **Observation-Based Rolling Correlations**: Calculates 7-observation and 14-observation rolling correlations strictly over valid news days.
7. **Sentiment Shock Event Detection**: Highlights trading sessions where sentiment deviates $\ge 2.0\sigma$ from the company mean and computes cumulative two-session forward returns.
8. **Signal Cross-Validation**: Compares FinBERT directional sentiment against GDELT tone, excluding missing GDELT tone records from the denominator.
9. **Deterministic RAG Query Routing**: Classifies queries into `TEMPORAL` (Chroma `$and` metadata filters), `AGGREGATE` (Predefined SQLite functions), and `SEMANTIC` (Top-5 vector similarity).
10. **Prompt Injection Defense & Provenance**: Treats retrieved headlines as untrusted data, prohibits execution of headline instructions, and cites full source provenance (`[1] Company — Date — Headline — URL`).

---

## 💻 Installation & Usage

### 1. Environment Setup
```bash
# Clone or navigate to the project directory
cd /Users/domaajay/.gemini/antigravity/scratch/financial-news-rag-analyst

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy `.env.example` to `.env` and set optional API credentials:
```bash
cp .env.example .env
```
Contents of `.env`:
```ini
LOG_LEVEL=INFO
ANALYSIS_WINDOW_DAYS=90
BATCH_SIZE=16

# Optional External LLM Credentials for RAG synthesis
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_hf_token_here
```
*Note: If no LLM keys are configured, the data pipeline and dashboard remain 100% functional, and the AI Analyst provides a controlled status message without crashing.*

### 3. Run Pipeline via CLI
```bash
# Execute full pipeline for default tickers (TCS.NS, INFY.NS, RELIANCE.NS, WIPRO.NS, HDFCBANK.NS)
python orchestrator.py --run-pipeline

# Run pipeline for a specific company over a 30-day window
python orchestrator.py --run-pipeline --company TCS.NS --days 30

# Seed synthetic demonstration data for immediate dashboard preview
python orchestrator.py --seed-demo
```

### 4. Launch Streamlit Dashboard
```bash
# Launch on canonical port 8501 (prevents duplicate instances)
./run_dashboard.sh
```
Access the dashboard in your web browser at `http://localhost:8501`.

### 5. Run Unit Test Suite
```bash
pytest -q
```

---

## 📊 Data Model

### SQLite Database (`data/finance.db`)
- **`headlines`**: Stores individual news articles (`article_id` PRIMARY KEY, `company`, `ticker`, `publication_date`, `headline`, `url`, `source_type` ('gdelt' | 'fallback'), `gdelt_tone`, `sentiment_score`, `sentiment_label`, `pipeline_run_id`).
- **`daily_metrics`**: Stores daily aggregated metrics (`PRIMARY KEY(company, date)`, `mean_sentiment`, `headline_count`, `has_news`, `daily_return`, `next_session_return`, `valid_sentiment`, `valid_return`, `rolling_7_observation_corr`, `rolling_14_observation_corr`).

### ChromaDB Collection (`data/chroma_db`)
- **Collection Name**: `financial_news`
- **Vector Embeddings**: 384-dimensional vectors from `sentence-transformers/all-MiniLM-L6-v2`.
- **Metadata**: `company`, `date`, `date_int` (YYYYMMDD integer), `sentiment_score`, `sentiment_label`, `url`, `source_type`.

---

## 📈 Statistical Methodology & Safety

1. **Correlation Metrics**: Pearson ($r$) and Spearman ($\rho$) correlations are computed per company. If valid paired observations $N < 15$, correlations return `NaN` with an `insufficient_observations` flag.
2. **Rolling Correlations**: Computed strictly over valid news observation windows (7-obs and 14-obs), requiring $\ge 5$ paired observations per window.
3. **Sentiment Shocks**: Identified when $|S_{\text{daily}} - \mu_{\text{comp}}| \ge 2.0 \cdot \sigma_{\text{comp}}$ (requires $\ge 20$ valid observations per company).
4. **Two-Session Forward Return**: Measures cumulative return over the next 2 available trading sessions ($Close_{t+2} / Close_t - 1.0$).
5. **Signal Cross-Validation**: Compares $\text{FinBERT} > 0$ vs $\text{GDELT Tone} > 0$. Fallback RSS articles missing `gdelt_tone` are excluded from the denominator.

---

## ⚠️ Important Methodological Limitations

1. **Correlation $\neq$ Causation**: Statistical association between headline sentiment and subsequent market return does not prove causal impact.
2. **Daily vs. Intraday Granularity**: Headlines published after market close are mapped to the next trading session; intraday sentiment shifts are aggregated daily.
3. **English-Language Bias**: News ingestion is restricted to English-language publications indexed by GDELT and Google News RSS.
4. **Syndication & Duplicates**: Despite deduplication, major wire releases may be referenced across multiple domains.
5. **Omitted Variables & Macro Factors**: Stock returns are driven by broader macroeconomic trends, interest rates, earnings reports, and market liquidity not fully captured in news headline sentiment alone.
6. **Sample Window**: The default 90-day window provides a short-term exploratory analysis rather than multi-year regime evaluation.

---

## 🛡️ Security & Prompt Injection Defense

- **Secret Protection**: API keys (`GROQ_API_KEY`, `HF_TOKEN`) are loaded via environment variables and never logged or displayed in UI components.
- **Untrusted Headline Content**: Retrieved news headlines are treated strictly as untrusted data within the system prompt. System instructions explicitly prohibit executing instructions embedded in headline text.
- **Zero Raw SQL Exposure**: The LLM is never permitted to generate or execute SQL queries. All database access uses parameterized, pre-defined functions in `store.py`.

---

## 📁 Project Structure

```
financial-news-rag-analyst/
├── config.py             # Centralized configuration & logging
├── fetch.py              # GDELT DOC 2.0 API & Google News RSS fetchers
├── transform.py          # FinBERT sentiment analysis & trading session alignment
├── store.py              # SQLite & ChromaDB dual-storage layer
├── analyze.py            # Statistical correlations, rolling windows & shock detection
├── llm_provider.py       # Groq & Hugging Face LLM provider abstractions
├── rag.py                # Deterministic RAG query router & evidence grounding
├── visualize.py          # Reusable Plotly chart functions
├── orchestrator.py       # End-to-end ingestion pipeline & CLI
├── app.py                # Streamlit 3-tab executive dashboard
├── verify_phase1.py      # Phase 1 verification script
├── verify_phase2.py      # Phase 2 verification script
├── verify_phase3.py      # Phase 3 verification script
├── requirements.txt      # Project dependencies
├── .env.example          # Environment settings template
├── .gitignore            # Git ignore rules
├── README.md             # Project documentation
└── tests/                # Pytest unit test suite
    ├── __init__.py
    ├── test_fetch.py
    ├── test_transform.py
    ├── test_store.py
    ├── test_analyze.py
    ├── test_rag.py
    └── test_orchestrator.py
```
