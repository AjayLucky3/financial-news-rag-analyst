"""
Configuration module for Financial News Sentiment + RAG Analyst.
Handles environment variables, ticker mappings, API endpoints, model parameters, and logging.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure offline mode for cached HuggingFace models to eliminate network retry delays
# Ensure HF_HOME points to user's real home cache if not explicitly set
if "HF_HOME" not in os.environ:
    user_hf = os.path.expanduser("~/.cache/huggingface")
    if os.path.isdir(user_hf):
        os.environ["HF_HOME"] = user_hf

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@dataclass
class Config:
    """Central configuration class for Phase 1 data pipeline."""

    # Default Tickers (NSE India listings)
    DEFAULT_TICKERS: List[str] = field(
        default_factory=lambda: [
            "TCS.NS",
            "INFY.NS",
            "RELIANCE.NS",
            "WIPRO.NS",
            "HDFCBANK.NS",
        ]
    )

    # Human-readable search terms for tickers to maximize GDELT & RSS relevance
    TICKER_QUERY_MAP: Dict[str, str] = field(
        default_factory=lambda: (
            __import__("companies").get_ticker_query_map()
            if os.path.exists(os.path.join(os.path.dirname(__file__), "companies.py"))
            else {
                "TCS.NS": '"Tata Consultancy Services" OR "TCS"',
                "INFY.NS": '"Infosys" OR "INFY"',
                "RELIANCE.NS": '"Reliance Industries" OR "Reliance"',
                "WIPRO.NS": '"Wipro"',
                "HDFCBANK.NS": '"HDFC Bank" OR "HDFCBANK"',
            }
        )
    )

    # Human-friendly display names for UI presentation
    COMPANY_DISPLAY_MAP: Dict[str, str] = field(
        default_factory=lambda: (
            __import__("companies").get_company_display_map()
            if os.path.exists(os.path.join(os.path.dirname(__file__), "companies.py"))
            else {
                "TCS.NS": "TCS",
                "INFY.NS": "Infosys",
                "RELIANCE.NS": "Reliance",
                "WIPRO.NS": "Wipro",
                "HDFCBANK.NS": "HDFC Bank",
            }
        )
    )

    # Analysis window configuration
    ANALYSIS_WINDOW_DAYS: int = int(os.getenv("ANALYSIS_WINDOW_DAYS", 90))

    # API Endpoints
    GDELT_DOC_API_URL: str = os.getenv(
        "GDELT_DOC_API_URL", "https://api.gdeltproject.org/api/v2/doc/doc"
    )
    GOOGLE_NEWS_RSS_URL: str = os.getenv(
        "GOOGLE_NEWS_RSS_URL", "https://news.google.com/rss/search"
    )

    # Network Request Settings
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", 15))
    GDELT_MAX_RETRIES: int = int(os.getenv("GDELT_MAX_RETRIES", 5))
    GDELT_BACKOFF_FACTOR: float = float(os.getenv("GDELT_BACKOFF_FACTOR", 2.0))

    # FinBERT Model Parameters
    MODEL_NAME: str = os.getenv("MODEL_NAME", "ProsusAI/finbert")
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", 16))
    MAX_SEQ_LENGTH: int = int(os.getenv("MAX_SEQ_LENGTH", 512))

    # RAM Budget Constraint (GB)
    RAM_BUDGET_GB: float = 12.0

    # LLM Settings
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    HF_MODEL: str = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", 20))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


def get_torch_device() -> str:
    """
    Selects the optimal torch device available (CUDA -> MPS -> CPU).
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def setup_logging(level: str = None) -> logging.Logger:
    """
    Configures standard logging format across all modules.
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("FinancialNewsRAG")
    logger.setLevel(numeric_level)
    return logger


def get_display_name(ticker: str) -> str:
    """Returns human-friendly company name for UI display."""
    if not ticker:
        return "UNKNOWN"
    if ticker in config.COMPANY_DISPLAY_MAP:
        return config.COMPANY_DISPLAY_MAP[ticker]
    try:
        import companies
        comp = companies.get_company(ticker)
        if comp:
            return comp.short_name
    except Exception:
        pass
    return ticker.replace(".NS", "")


# Instantiate default config instance
config = Config()
logger = setup_logging(config.LOG_LEVEL)
