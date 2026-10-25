"""
Unit tests for store.py database storage layer.
Verifies SQLite schema, indexes, idempotency, Chroma vector insertion, and metadata filtering.
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import store


@pytest.fixture(autouse=True)
def clean_test_databases(tmp_path):
    """Fixture to ensure clean DB for each test."""
    store.set_storage_mode(demo=False)
    test_db = tmp_path / "test_finance.db"
    test_chroma = tmp_path / "test_chroma"

    store.SQLITE_DB_PATH = test_db
    store.CHROMA_DB_PATH = test_chroma

    store.initialize_database()
    yield
    store.set_storage_mode(demo=False)
    if os.path.exists(test_db):
        os.remove(test_db)


def test_sqlite_schema_and_indexes():
    """Verify tables and indexes created properly in SQLite."""
    conn = sqlite3.connect(store.SQLITE_DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cur.fetchall()]
    assert "headlines" in tables
    assert "daily_metrics" in tables

    cur.execute("SELECT name FROM sqlite_master WHERE type='index';")
    indexes = [r[0] for r in cur.fetchall()]
    assert "idx_headlines_company" in indexes
    assert "idx_headlines_pubdate" in indexes
    conn.close()


def test_idempotent_headline_writes():
    """Verify duplicate headline inserts do not cause duplicate records."""
    df = pd.DataFrame([
        {
            "company": "TCS.NS",
            "ticker": "TCS.NS",
            "timestamp": "2026-09-20 10:00:00",
            "title": "TCS Cloud Expansion",
            "url": "https://example.com/tcs",
            "source": "GDELT",
            "sentiment_score": 0.8,
        }
    ])

    store.save_headlines(df)
    store.save_headlines(df)  # Duplicate write

    conn = sqlite3.connect(store.SQLITE_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM headlines;")
    count = cur.fetchone()[0]
    conn.close()

    assert count == 1


def test_chroma_insertion_and_filter():
    """Verify document insertion and compound filter retrieval in Chroma."""
    df = pd.DataFrame([
        {
            "company": "INFY.NS",
            "ticker": "INFY.NS",
            "timestamp": "2026-09-21 11:00:00",
            "title": "Infosys Q2 margin analysis",
            "url": "https://example.com/infy",
            "source": "GoogleNewsRSS",
            "sentiment_score": -0.5,
        }
    ])

    store.save_to_chroma(df)
    coll = store.get_vector_collection()
    assert coll.count() == 1

    res = coll.query(
        query_texts=["margin analysis"],
        n_results=1,
        where={"$and": [{"company": {"$eq": "INFY.NS"}}, {"source_type": {"$eq": "fallback"}}]}
    )
    assert len(res["documents"][0]) == 1
    assert "Infosys" in res["documents"][0][0]


def test_market_price_columns_persisted():
    """Verify market price columns (close, open, high, low, volume) are saved to SQLite and loaded into DataFrame."""
    metrics_df = pd.DataFrame([
        {
            "company": "TCS.NS",
            "ticker": "TCS.NS",
            "date": "2026-09-22",
            "close": 3500.50,
            "open": 3480.00,
            "high": 3520.00,
            "low": 3475.00,
            "volume": 1200000,
            "sentiment_score": 0.4,
            "headline_count": 3,
            "same_day_return": 0.005,
            "next_day_return": 0.012,
        }
    ])

    store.save_daily_metrics(metrics_df)
    retrieved_df = store.get_daily_metrics_records("TCS.NS")

    assert not retrieved_df.empty
    assert "close" in retrieved_df.columns
    assert "Close" in retrieved_df.columns
    assert float(retrieved_df.iloc[0]["close"]) == 3500.50
    assert float(retrieved_df.iloc[0]["Close"]) == 3500.50


def test_storage_mode_path_isolation():
    """Verify REAL and DEMO storage modes target different paths and collections."""
    store.set_storage_mode(demo=False)
    assert store.get_storage_mode() == "REAL"
    assert "demo" not in str(store.get_sqlite_path()).lower()
    assert store.get_chroma_collection_name() == "financial_news"

    store.set_storage_mode(demo=True)
    assert store.get_storage_mode() == "DEMO"
    assert "demo_finance.db" in str(store.get_sqlite_path())
    assert store.get_chroma_collection_name() == "financial_news_demo"

    store.set_storage_mode(demo=False)


def test_no_demo_urls_in_real_mode():
    """Verify that in REAL mode, save_to_chroma does not persist demo.example.com URLs."""
    store.set_storage_mode(demo=False)
    df_real = pd.DataFrame([
        {
            "company": "TCS.NS",
            "ticker": "TCS.NS",
            "timestamp": "2026-09-25 10:00:00",
            "title": "TCS AI Partnership Announcement",
            "url": "https://finance.yahoo.com/news/tcs-ai-partnership-1234.html",
            "source": "GoogleNewsRSS",
            "sentiment_score": 0.65,
        }
    ])
    store.save_to_chroma(df_real)
    coll = store.get_vector_collection()
    res = coll.get()
    urls = [m.get("url", "") for m in res.get("metadatas", [])]
    for url in urls:
        assert "demo.example.com" not in url


def test_1970_date_regression():
    """Verify invalid or zero-epoch timestamps are sanitized and never become 1970-01-01."""
    store.set_storage_mode(demo=False)
    df_bad_date = pd.DataFrame([
        {
            "company": "INFY.NS",
            "ticker": "INFY.NS",
            "timestamp": "1970-01-01 00:00:00", # Epoch 0 / invalid timestamp!
            "title": "Infosys quarterly update with invalid date",
            "url": "https://example.com/infy_bad_date",
            "source": "GoogleNewsRSS",
            "sentiment_score": 0.1,
        }
    ])
    store.save_to_chroma(df_bad_date)
    coll = store.get_vector_collection()
    res = coll.get()
    dates = [m.get("date", "") for m in res.get("metadatas", [])]
    for d_str in dates:
        assert not d_str.startswith("1970")

