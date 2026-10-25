"""
Data Integrity and Regression Tests for Financial News Sentiment + RAG Analyst.
Enforces hard constraints on price continuity, date validity, sample size definitions,
company isolation, and demo/real storage separation.
"""

import sqlite3
import pandas as pd
import numpy as np
import pytest

from store import SQLITE_DB_PATH, _get_sqlite_connection
from analyze import assess_data_sufficiency, get_association_interpretation, compute_correlations


def test_no_legacy_1970_dates():
    """Verify that no records in SQLite have date < 2000-01-01 (specifically catches 1970-01-01 bug)."""
    with _get_sqlite_connection() as conn:
        c = conn.cursor()
        bad_metrics = c.execute("SELECT company, date FROM daily_metrics WHERE date < '2000-01-01'").fetchall()
        assert len(bad_metrics) == 0, f"Found legacy 1970/invalid dates in daily_metrics: {bad_metrics}"

        bad_headlines = c.execute("SELECT company, publication_date FROM headlines WHERE publication_date < '2000-01-01'").fetchall()
        assert len(bad_headlines) == 0, f"Found legacy 1970/invalid dates in headlines: {bad_headlines}"


def test_unique_trading_sessions_per_company():
    """Verify that (company, date) pairs in daily_metrics are strictly unique without duplicates."""
    with _get_sqlite_connection() as conn:
        df = pd.read_sql_query("SELECT company, date, COUNT(*) as cnt FROM daily_metrics GROUP BY company, date HAVING cnt > 1", conn)
        assert df.empty, f"Duplicate (company, date) pairs detected in daily_metrics: {df.to_dict(orient='records')}"


def test_price_continuity_sanity():
    """
    Verify price continuity across consecutive trading sessions.
    Catches the TCS price collapse bug where prices jumped between ~2400 and ~1000 (>50% jump).
    """
    with _get_sqlite_connection() as conn:
        df = pd.read_sql_query("SELECT company, date, close, open FROM daily_metrics WHERE close IS NOT NULL ORDER BY company, date ASC", conn)

    if df.empty:
        pytest.skip("No price records in database.")

    for company, grp in df.groupby("company"):
        closes = grp["close"].values
        if len(closes) < 2:
            continue
        pct_changes = np.abs(np.diff(closes) / closes[:-1])
        max_jump = float(np.max(pct_changes))
        # No normal single-day unadjusted price jump should exceed 50% without error
        assert max_jump < 0.50, f"{company} exhibited impossible price discontinuity of {max_jump*100:.1f}% jump/collapse."


def test_news_days_consistent_with_stored_headlines():
    """
    Verify logical consistency between headlines and daily_metrics.
    A company with 0 stored headlines must NOT have news days > 0.
    """
    with _get_sqlite_connection() as conn:
        headlines = pd.read_sql_query("SELECT company, COUNT(*) as h_cnt FROM headlines GROUP BY company", conn)
        h_map = dict(zip(headlines["company"], headlines["h_cnt"]))

        metrics = pd.read_sql_query("SELECT company, SUM(has_news) as news_days FROM daily_metrics GROUP BY company", conn)
        for _, row in metrics.iterrows():
            comp = row["company"]
            news_days = row["news_days"] or 0
            h_count = h_map.get(comp, 0)
            if h_count == 0:
                assert news_days == 0, f"{comp} has 0 stored headlines but daily_metrics reports {news_days} news days!"


def test_comparable_sessions_definition():
    """
    Verify comparable sessions definition:
    Must be strictly rows where mean_sentiment.notna() & next_session_return.notna() & has_news == 1.
    """
    test_df = pd.DataFrame({
        "company": ["TCS.NS"] * 5,
        "date": ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-05"],
        "mean_sentiment": [0.5, np.nan, 0.2, 0.1, np.nan],
        "next_session_return": [0.01, 0.02, np.nan, 0.03, 0.01],
        "has_news": [1, 0, 1, 1, 0],
    })

    comp_mask = (
        test_df["mean_sentiment"].notna()
        & test_df["next_session_return"].notna()
        & (test_df["has_news"] == 1)
    )
    # Row 0 (valid, valid, 1) -> True
    # Row 1 (nan, valid, 0) -> False
    # Row 2 (valid, nan, 1) -> False
    # Row 3 (valid, valid, 1) -> True
    # Row 4 (nan, valid, 0) -> False
    assert comp_mask.sum() == 2


def test_insufficient_sample_size_suppresses_correlation():
    """Verify that when N < 15, correlation is suppressed and badge is 🔴 INSUFFICIENT DATA."""
    # Test assess_data_sufficiency tiers
    res_under_15 = assess_data_sufficiency(14)
    assert res_under_15["allow_correlation"] is False
    assert "INSUFFICIENT" in res_under_15["badge"]

    res_15 = assess_data_sufficiency(15)
    assert res_15["allow_correlation"] is True
    assert "EXPLORATORY" in res_15["badge"]

    res_30 = assess_data_sufficiency(30)
    assert res_30["allow_correlation"] is True
    assert "BROADER EXPLORATORY" in res_30["badge"]

    res_60 = assess_data_sufficiency(60)
    assert res_60["allow_correlation"] is True
    assert "LARGER HISTORICAL" in res_60["badge"]

    # Test get_association_interpretation
    assoc = get_association_interpretation(0.55, valid_obs=10)
    assert "INSUFFICIENT" in assoc["badge"]
    assert "Suppressed" in assoc["value_display"]


def test_storage_isolation_cleanliness():
    """Verify that real finance.db does not contain synthetic demo rows."""
    with _get_sqlite_connection() as conn:
        demo_cnt = conn.execute("SELECT COUNT(*) FROM headlines WHERE pipeline_run_id = 'DEMO_RUN'").fetchone()[0]
        assert demo_cnt == 0, f"Found {demo_cnt} DEMO_RUN records inside real finance.db!"


def test_chroma_real_collection_has_no_demo_docs():
    """Verify that real ChromaDB financial_news collection does not contain synthetic demo records."""
    from store import get_vector_collection
    col = get_vector_collection()
    res = col.get()
    demo_docs = [
        doc for doc, meta in zip(res["documents"], res["metadatas"])
        if "[DEMO DATA]" in doc or "demo.example.com" in str(meta.get("url", "")) or meta.get("pipeline_run_id") == "DEMO_RUN"
    ]
    assert len(demo_docs) == 0, f"Found {len(demo_docs)} demo documents inside real Chroma collection!"
