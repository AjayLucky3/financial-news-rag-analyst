"""
Unit tests for fetch.py data retrieval functions.
Uses mocks to test network error handling, GDELT backoff, JSON validation, and RSS fallback.
"""

import os
import sys
from datetime import datetime
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fetch import (
    fetch_gdelt_news_raw,
    fetch_gdelt_news,
    fetch_google_news_rss,
    fetch_news_for_ticker,
    fetch_financial_data,
    GDELTRateLimitError,
    GDELTServerError,
    GDELTFetchError,
)


@patch("fetch.requests.get")
def test_gdelt_raw_validation_success(mock_get):
    """Test successful GDELT raw response validation."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"Content-Type": "application/json"}
    mock_resp.text = '{"articles": [{"title": "TCS Growth", "seendate": "20260920T100000Z"}]}'
    mock_resp.json.return_value = {"articles": [{"title": "TCS Growth", "seendate": "20260920T100000Z"}]}
    mock_get.return_value = mock_resp

    data = fetch_gdelt_news_raw("TCS", "20260901000000", "20260920000000")
    assert "articles" in data
    assert len(data["articles"]) == 1


@patch("fetch.requests.get")
def test_gdelt_raw_429_rate_limit(mock_get):
    """Test GDELT HTTP 429 raises GDELTRateLimitError."""
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_get.return_value = mock_resp

    with pytest.raises(GDELTRateLimitError):
        # Call directly bypassing tenacity retry decorator for unit test
        fetch_gdelt_news_raw.__wrapped__("TCS", "20260901000000", "20260920000000")


@patch("fetch.requests.get")
def test_gdelt_raw_5xx_server_error(mock_get):
    """Test GDELT HTTP 500 raises GDELTServerError."""
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_get.return_value = mock_resp

    with pytest.raises(GDELTServerError):
        fetch_gdelt_news_raw.__wrapped__("TCS", "20260901000000", "20260920000000")


@patch("fetch.fetch_gdelt_news_raw")
@patch("fetch.fetch_google_news_rss")
def test_fetch_news_for_ticker_fallback(mock_rss, mock_gdelt):
    """Test fallback to Google News RSS when GDELT returns 0 results."""
    mock_gdelt.return_value = pd.DataFrame()
    mock_rss.return_value = pd.DataFrame([
        {
            "timestamp": pd.Timestamp("2026-09-20"),
            "title": "Infosys RSS Headline",
            "url": "https://news.google.com/item1",
            "domain": "news.google.com",
            "source": "GoogleNewsRSS",
        }
    ])

    start_dt = datetime(2026, 9, 1)
    end_dt = datetime(2026, 9, 20)
    df = fetch_news_for_ticker("INFY.NS", start_dt, end_dt)

    assert not df.empty
    assert df.iloc[0]["source"] == "GoogleNewsRSS"
    assert df.iloc[0]["ticker"] == "INFY.NS"


@patch("fetch.yf.Ticker")
def test_fetch_financial_data_empty(mock_ticker):
    """Test empty yfinance history returns empty mapping entry gracefully."""
    mock_instance = MagicMock()
    mock_instance.history.return_value = pd.DataFrame()
    mock_ticker.return_value = mock_instance

    data = fetch_financial_data(["INVALID.NS"], datetime(2026, 9, 1), datetime(2026, 9, 20))
    assert "INVALID.NS" not in data
