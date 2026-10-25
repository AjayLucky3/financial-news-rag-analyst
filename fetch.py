"""
Data fetching module for Financial News Sentiment + RAG Analyst.
Handles GDELT DOC 2.0 API fetching with backoff and validation,
Google News RSS fallback fetching, and yfinance price history fetching.
"""

import time
import urllib.parse
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests
import feedparser
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    retry_if_exception_type,
)

from config import config, logger


class GDELTFetchError(Exception):
    """Custom exception raised when GDELT API returns unrecoverable error."""

    pass


class GDELTRateLimitError(GDELTFetchError):
    """Exception raised when GDELT returns 429 Too Many Requests."""

    pass


class GDELTServerError(GDELTFetchError):
    """Exception raised when GDELT returns 5xx server errors."""

    pass


def _should_retry_gdelt_exception(exc: Exception) -> bool:
    """Predicate function to retry on rate limits, server errors, or connection issues."""
    if isinstance(exc, (GDELTRateLimitError, GDELTServerError, requests.RequestException)):
        return True
    return False


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=32),
    retry=retry_if_exception(_should_retry_gdelt_exception),
    reraise=True,
)
def fetch_gdelt_news_raw(
    query: str, start_datetime_str: str, end_datetime_str: str, max_records: int = 250
) -> dict:
    """
    Performs raw network call to GDELT DOC 2.0 API with exponential backoff for 429 and 5xx errors.
    Validates HTTP status code and content-type before returning parsed JSON.

    Dates must be in YYYYMMDDHHMMSS format.
    """
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "startdatetime": start_datetime_str,
        "enddatetime": end_datetime_str,
        "maxrecords": str(max_records),
    }

    logger.debug(f"Fetching GDELT data with params: {params}")

    response = requests.get(
        config.GDELT_DOC_API_URL, params=params, timeout=config.REQUEST_TIMEOUT
    )

    # Validate HTTP response status codes
    if response.status_code == 429:
        logger.warning("GDELT HTTP 429: Rate limited. Retrying with exponential backoff...")
        raise GDELTRateLimitError("GDELT API HTTP 429 Rate Limit Exceeded")

    if response.status_code >= 500:
        logger.warning(
            f"GDELT HTTP {response.status_code}: Server error. Retrying with backoff..."
        )
        raise GDELTServerError(f"GDELT API HTTP {response.status_code} Server Error")

    if response.status_code != 200:
        raise GDELTFetchError(
            f"GDELT API returned HTTP {response.status_code}: {response.text[:200]}"
        )

    # Validate content before JSON parsing
    content_type = response.headers.get("Content-Type", "")
    text_content = response.text.strip()

    if not text_content:
        logger.info("GDELT API returned empty response body.")
        return {"articles": []}

    # GDELT occasionally returns HTML error pages with HTTP 200 status code
    if text_content.startswith("<") or "html" in content_type.lower():
        logger.warning(
            "GDELT API returned HTML instead of JSON. Assuming no results or temporary block."
        )
        return {"articles": []}

    try:
        data = response.json()
        return data
    except Exception as e:
        logger.error(f"Failed to parse GDELT response as JSON: {e}")
        raise GDELTFetchError(f"Invalid JSON from GDELT: {e}")


def fetch_gdelt_news(
    query: str,
    start_date: datetime,
    end_date: datetime,
    max_records: int = 250,
) -> pd.DataFrame:
    """
    Fetches news articles from GDELT DOC 2.0 API between start_date and end_date.

    Parameters:
        query: GDELT search query string.
        start_date: start datetime object.
        end_date: end datetime object.
        max_records: maximum number of records to retrieve (max 250).

    Returns:
        pd.DataFrame with columns: ['timestamp', 'title', 'url', 'domain', 'source']
    """
    # Enforce strict YYYYMMDDHHMMSS format for GDELT API date parameters
    start_str = start_date.strftime("%Y%m%d%H%M%S")
    end_str = end_date.strftime("%Y%m%d%H%M%S")

    try:
        raw_data = fetch_gdelt_news_raw(
            query=query,
            start_datetime_str=start_str,
            end_datetime_str=end_str,
            max_records=max_records,
        )

        articles = raw_data.get("articles", [])
        if not articles:
            logger.info(f"No GDELT articles found for query: '{query}'")
            return pd.DataFrame(columns=["timestamp", "title", "url", "domain", "source"])

        records = []
        for art in articles:
            seendate = art.get("seendate", "")
            title = art.get("title", "").strip()
            url = art.get("url", "")
            domain = art.get("domain", "")

            # Parse seendate e.g. '20260928T120000Z' or '20260928120000'
            dt = None
            if seendate:
                clean_date_str = (
                    seendate.replace("T", "").replace("Z", "").replace("-", "").replace(":", "")
                )
                try:
                    dt = datetime.strptime(clean_date_str[:14], "%Y%m%d%H%M%S")
                except ValueError:
                    dt = None

            if dt and title:
                records.append(
                    {
                        "timestamp": pd.to_datetime(dt),
                        "title": title,
                        "url": url,
                        "domain": domain,
                        "source": "GDELT",
                    }
                )

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)
        return df

    except GDELTFetchError as e:
        logger.warning(f"GDELT fetch failed for query '{query}': {e}")
        return pd.DataFrame(columns=["timestamp", "title", "url", "domain", "source"])


def fetch_google_news_rss(query: str, max_records: int = 50) -> pd.DataFrame:
    """
    Fallback news fetcher using Google News RSS via feedparser and BeautifulSoup.

    Parameters:
        query: Search query string.
        max_records: Maximum number of entries to extract.

    Returns:
        pd.DataFrame with columns: ['timestamp', 'title', 'url', 'domain', 'source']
    """
    encoded_query = urllib.parse.quote(query)
    rss_url = f"{config.GOOGLE_NEWS_RSS_URL}?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

    logger.info(f"Fetching Google News RSS fallback for query: '{query}'")
    try:
        feed = feedparser.parse(rss_url)
        entries = feed.entries[:max_records]

        records = []
        for entry in entries:
            title = entry.get("title", "").strip()
            link = entry.get("link", "")
            published = entry.get("published", "")

            # Clean summary text if available
            summary_raw = entry.get("summary", "")
            cleaned_summary = ""
            if summary_raw:
                cleaned_summary = BeautifulSoup(summary_raw, "html.parser").get_text()

            # Parse published date string
            dt = None
            if published:
                try:
                    dt = pd.to_datetime(published)
                    if dt is not None and hasattr(dt, "tz_localize") and dt.tz is not None:
                        dt = dt.tz_convert(None)
                    if pd.isna(dt) or dt.year < 2000:
                        dt = pd.Timestamp.now()
                except Exception:
                    dt = pd.Timestamp.now()
            else:
                dt = pd.Timestamp.now()

            domain = urllib.parse.urlparse(link).netloc

            if title:
                records.append(
                    {
                        "timestamp": pd.to_datetime(dt),
                        "title": title,
                        "url": link,
                        "domain": domain,
                        "source": "GoogleNewsRSS",
                    }
                )

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)
        return df

    except Exception as e:
        logger.error(f"Failed to fetch Google News RSS for query '{query}': {e}")
        return pd.DataFrame(columns=["timestamp", "title", "url", "domain", "source"])


def fetch_news_for_ticker(
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    query_override: Optional[str] = None,
) -> pd.DataFrame:
    """
    Primary news retrieval workflow for a ticker:
    1. Query GDELT DOC 2.0 API first.
    2. If GDELT fails or returns 0 articles, fallback to Google News RSS.

    Parameters:
        ticker: Ticker symbol (e.g. 'TCS.NS').
        start_date: Start datetime object.
        end_date: End datetime object.
        query_override: Optional query string override.

    Returns:
        pd.DataFrame containing news items for the ticker.
    """
    query = query_override or config.TICKER_QUERY_MAP.get(ticker, ticker)
    logger.info(f"Fetching news for ticker: {ticker} (Query: '{query}')")

    # Primary: GDELT
    df = fetch_gdelt_news(query=query, start_date=start_date, end_date=end_date)

    # Fallback: Google News RSS
    if df.empty:
        logger.warning(
            f"GDELT returned 0 news items for {ticker}. Initiating Google News RSS fallback..."
        )
        df = fetch_google_news_rss(query=query)

    df["ticker"] = ticker
    return df


def fetch_financial_data(
    tickers: List[str], start_date: datetime, end_date: datetime
) -> Dict[str, pd.DataFrame]:
    """
    Fetches historical price data for each ticker sequentially using yfinance.

    Hard constraint: Tickers are processed sequentially to avoid rate limits / concurrency issues.

    Parameters:
        tickers: List of ticker symbols (e.g. ['TCS.NS', 'INFY.NS'])
        start_date: Start date for yfinance history.
        end_date: End date for yfinance history.

    Returns:
        Dict mapping ticker symbol to price DataFrame with DatetimeIndex and standard price columns.
    """
    price_data = {}

    for ticker in tickers:
        logger.info(f"Sequentially fetching yfinance data for {ticker}...")
        try:
            yticker = yf.Ticker(ticker)
            df = yticker.history(start=start_date, end=end_date)

            if df.empty:
                logger.warning(f"yfinance returned empty dataset for ticker {ticker}")
                continue

            # Ensure DatetimeIndex is localized cleanly and index is named 'date'
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)

            # Strip time part to ensure pure date index representing trading sessions
            df.index = pd.to_datetime(df.index.date)
            df.index.name = "date"

            # Retain standard required columns
            required_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
            df = df[required_cols]

            price_data[ticker] = df
            logger.info(
                f"Successfully retrieved {len(df)} trading sessions for {ticker} "
                f"from {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}."
            )

        except Exception as e:
            logger.error(f"Failed to fetch yfinance data for {ticker}: {e}")

    return price_data
