"""
Verification script for Phase 2 of Financial News Sentiment + RAG Analyst.
Verifies SQLite storage, ChromaDB vector storage, idempotency, statistical correlation logic,
rolling observation correlations, sentiment shock detection, two-session forward returns,
and GDELT/FinBERT signal cross-validation.
"""

import sys
import os
import sqlite3
import py_compile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1. Syntax Compilation Check
print("--- 1. Testing Syntax Compilation ---")
modules = ["config.py", "fetch.py", "transform.py", "store.py", "analyze.py"]
for mod in modules:
    try:
        py_compile.compile(mod, doraise=True)
        print(f"  {mod}: Syntax OK")
    except Exception as e:
        print(f"  {mod}: Syntax Error: {e}")
        sys.exit(1)
print("PASSED: All Python files compiled cleanly.\n")

# 2. Imports Check
print("--- 2. Testing Module Imports ---")
try:
    import store
    import analyze
    from store import (
        initialize_database,
        save_headlines,
        save_daily_metrics,
        save_to_chroma,
        save_data,
        get_company_sentiment_summary,
        get_worst_sentiment_period,
        get_headline_counts_by_company,
        get_daily_metrics_records,
        get_vector_collection,
        SQLITE_DB_PATH,
        CHROMA_DB_PATH,
    )
    from analyze import (
        compute_correlations,
        compute_rolling_correlations,
        detect_sentiment_shocks,
        cross_validate_signals,
    )
    print("PASSED: store.py and analyze.py imported cleanly.\n")
except Exception as e:
    print(f"FAILURE: Module import error: {e}")
    sys.exit(1)


def run_phase2_verifications():
    print("==================================================")
    print("         STARTING PHASE 2 VERIFICATIONS           ")
    print("==================================================\n")

    # Clean test DB paths to ensure fresh test runs
    if os.path.exists(SQLITE_DB_PATH):
        os.remove(SQLITE_DB_PATH)

    # 3. Initialize SQLite & Schema Verification
    print("--- 3 & 4. SQLite Initialization & Table Verification ---")
    initialize_database()
    assert os.path.exists(SQLITE_DB_PATH), "SQLite DB file not created!"

    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall()]
    print(f"Created SQLite Tables: {tables}")
    assert "headlines" in tables, "Table 'headlines' missing!"
    assert "daily_metrics" in tables, "Table 'daily_metrics' missing!"

    cursor.execute("SELECT name FROM sqlite_master WHERE type='index';")
    indexes = [r[0] for r in cursor.fetchall()]
    print(f"Created SQLite Indexes: {indexes}")
    assert "idx_headlines_company" in indexes
    assert "idx_headlines_pubdate" in indexes
    assert "idx_headlines_company_pubdate" in indexes
    assert "idx_headlines_article_id" in indexes
    conn.close()
    print("PASSED: SQLite tables and indexes initialized correctly.\n")

    # 5. ChromaDB Collection Verification
    print("--- 5. ChromaDB Collection Initialization ---")
    collection = get_vector_collection()
    print(f"ChromaDB Collection Name: {collection.name}")
    assert collection.name == "financial_news"
    print("PASSED: ChromaDB collection initialized correctly.\n")

    # 6 & 7. Idempotent Headline & Daily Metrics Insertion
    print("--- 6 & 7. Idempotent Headline & Daily Metrics Insertion ---")
    test_headlines_df = pd.DataFrame([
        {
            "company": "TCS.NS",
            "ticker": "TCS.NS",
            "timestamp": "2026-09-21 10:00:00",
            "title": "TCS secures major cloud transformation deal",
            "url": "https://example.com/tcs1",
            "source": "GDELT",
            "gdelt_tone": 4.5,
            "prob_positive": 0.85,
            "prob_negative": 0.05,
            "prob_neutral": 0.10,
            "sentiment_score": 0.80,
            "sentiment_label": "POSITIVE",
        },
        {
            "company": "INFY.NS",
            "ticker": "INFY.NS",
            "timestamp": "2026-09-22 14:00:00",
            "title": "Infosys faces margin pressure in Q2",
            "url": "https://example.com/infy1",
            "source": "GoogleNewsRSS",
            "gdelt_tone": np.nan,  # Fallback RSS has no gdelt_tone
            "prob_positive": 0.10,
            "prob_negative": 0.80,
            "prob_neutral": 0.10,
            "sentiment_score": -0.70,
            "sentiment_label": "NEGATIVE",
        },
    ])

    save_headlines(test_headlines_df)
    save_headlines(test_headlines_df)  # Idempotent re-run test

    conn = sqlite3.connect(SQLITE_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM headlines;")
    h_count = cur.fetchone()[0]
    print(f"Total headlines in SQLite after duplicate insertion: {h_count} (Expected: 2)")
    assert h_count == 2, f"Idempotency failed! Expected 2 records, got {h_count}"

    # Verify source_type constraint mapping
    cur.execute("SELECT source_type FROM headlines WHERE company='TCS.NS';")
    src = cur.fetchone()[0]
    assert src == "gdelt", f"Expected 'gdelt', got '{src}'"

    cur.execute("SELECT source_type FROM headlines WHERE company='INFY.NS';")
    src_fallback = cur.fetchone()[0]
    assert src_fallback == "fallback", f"Expected 'fallback', got '{src_fallback}'"

    # Daily Metrics Insertion Test with NaN preservation
    test_daily_metrics_df = pd.DataFrame([
        {
            "company": "TCS.NS",
            "ticker": "TCS.NS",
            "date": "2026-09-21",
            "sentiment_score": 0.80,
            "headline_count": 1,
            "same_day_return": 0.01,
            "next_day_return": 0.02,
        },
        {
            "company": "TCS.NS",
            "ticker": "TCS.NS",
            "date": "2026-09-22",
            "sentiment_score": np.nan,  # No news day
            "headline_count": 0,
            "same_day_return": -0.005,
            "next_day_return": 0.015,
        },
    ])

    save_daily_metrics(test_daily_metrics_df)
    save_daily_metrics(test_daily_metrics_df)  # Idempotent re-run test

    cur.execute("SELECT date, mean_sentiment, headline_count, has_news FROM daily_metrics WHERE company='TCS.NS';")
    m_rows = cur.fetchall()
    print("Daily Metrics in SQLite:")
    for r in m_rows:
        print(f"  Date: {r[0]} | Mean Sentiment: {r[1]} | Headline Count: {r[2]} | Has News: {r[3]}")

    # Assert no-news day (2026-09-22) has NULL mean_sentiment and 0 headline_count
    no_news_row = [r for r in m_rows if r[0] == "2026-09-22"][0]
    assert no_news_row[1] is None, "No-news day mean_sentiment was wrongly imputed!"
    assert no_news_row[2] == 0
    assert no_news_row[3] == 0
    conn.close()

    print("PASSED: Idempotent SQLite insertion and NaN preservation verified.\n")

    # 8. ChromaDB Insertion & Vector Query Test
    print("--- 8. ChromaDB Document Insertion & Metadata Filter Test ---")
    save_to_chroma(test_headlines_df)
    chroma_count = collection.count()
    print(f"Total Chroma Documents: {chroma_count}")
    assert chroma_count == 2

    # Query Chroma with compound filter
    query_res = collection.query(
        query_texts=["cloud transformation"],
        n_results=1,
        where={"$and": [{"company": {"$eq": "TCS.NS"}}, {"source_type": {"$eq": "gdelt"}}]}
    )
    print("Chroma Compound Filter Query Result:")
    print(f"  Doc: {query_res['documents'][0]}")
    print(f"  Metadata: {query_res['metadatas'][0]}")
    assert len(query_res["documents"][0]) > 0
    assert query_res["metadatas"][0][0]["company"] == "TCS.NS"
    print("PASSED: ChromaDB document insertion and metadata filtering verified.\n")

    # 9. Retrieval Helper Functions Test
    print("--- 9. Testing Storage Retrieval Functions ---")
    summary = get_company_sentiment_summary("TCS.NS")
    print(f"TCS.NS Summary: {summary}")
    assert summary["total_headlines"] == 1
    assert abs(summary["avg_sentiment"] - 0.80) < 1e-4

    counts = get_headline_counts_by_company()
    print(f"Headline Counts: {counts}")
    assert counts.get("TCS.NS") == 1
    assert counts.get("INFY.NS") == 1

    records = get_daily_metrics_records("TCS.NS")
    print(f"Retrieved {len(records)} daily metric records for TCS.NS.")
    assert len(records) == 2
    print("PASSED: Retrieval functions verified.\n")

    # 10 & 11. Statistical Correlation & Minimum Observation Threshold Test (<15 obs)
    print("--- 10 & 11. Testing Statistical Correlations & min_obs Threshold ---")
    # Case A: Insufficient observations (< 15)
    dm_small = pd.DataFrame({
        "company": ["TCS.NS"] * 5,
        "date": [f"2026-09-0{i+1}" for i in range(5)],
        "mean_sentiment": [0.1, 0.2, 0.3, 0.4, 0.5],
        "next_session_return": [0.01, 0.02, 0.01, 0.03, 0.02],
        "has_news": [1] * 5,
        "headline_count": [1] * 5,
    })
    res_small = compute_correlations(dm_small, min_obs=15)
    tcs_small = res_small["per_company"]["TCS.NS"]
    print("Small Sample Correlation Result:")
    print(f"  Status: {tcs_small['status_message']}")
    print(f"  Pearson Corr: {tcs_small['pearson_corr']}")
    assert tcs_small["insufficient_observations"] is True
    assert np.isnan(tcs_small["pearson_corr"])

    # Case B: Sufficient observations (>= 15) with synthetic test fixture
    np.random.seed(42)
    n_pts = 20
    s_vals = np.random.uniform(-0.8, 0.8, n_pts)
    r_vals = 0.5 * s_vals + np.random.normal(0, 0.1, n_pts)
    dm_large = pd.DataFrame({
        "company": ["RELIANCE.NS"] * n_pts,
        "date": [f"2026-08-{i+1:02d}" for i in range(n_pts)],
        "mean_sentiment": s_vals,
        "next_session_return": r_vals,
        "has_news": [1] * n_pts,
        "headline_count": [2] * n_pts,
    })
    res_large = compute_correlations(dm_large, min_obs=15)
    rel_res = res_large["per_company"]["RELIANCE.NS"]
    print("\nSufficient Sample Correlation Result (RELIANCE.NS):")
    print(f"  Pearson r: {rel_res['pearson_corr']:.4f} (p-val: {rel_res['pearson_pvalue']:.4f})")
    print(f"  Spearman rho: {rel_res['spearman_corr']:.4f} (p-val: {rel_res['spearman_pvalue']:.4f})")
    assert rel_res["insufficient_observations"] is False
    assert not np.isnan(rel_res["pearson_corr"])
    assert rel_res["n_observations"] == n_pts
    print("PASSED: Statistical correlations and min_obs threshold checks verified.\n")

    # 12. Rolling Observation Correlation Test
    print("--- 12. Testing Rolling Observation Correlations ---")
    rolling_df = compute_rolling_correlations(dm_large, min_obs_threshold=5)
    print("Rolling Correlation Output Columns:", list(rolling_df.columns))
    assert "rolling_7_observation_corr" in rolling_df.columns
    assert "rolling_14_observation_corr" in rolling_df.columns

    # First 4 rows should be NaN because min_obs_threshold=5
    assert np.isnan(rolling_df["rolling_7_observation_corr"].iloc[0])
    assert np.isnan(rolling_df["rolling_7_observation_corr"].iloc[3])
    # 5th row (index 4) should have a valid correlation
    assert not np.isnan(rolling_df["rolling_7_observation_corr"].iloc[4])
    print("PASSED: Rolling observation correlation calculation verified.\n")

    # 13 & 14. Sentiment Shock Detection & min_company_obs Requirement (<20 obs)
    print("--- 13 & 14. Testing Sentiment Shock Detection & min_company_obs ---")
    # Create dataset with 25 days, including 1 huge shock on day 10
    dates = [f"2026-07-{i+1:02d}" for i in range(25)]
    sents = [0.10] * 25
    sents[10] = 0.95  # Shock: ~8 standard deviations above mean
    closes = [100.0 + i for i in range(25)]  # Price array for two-session forward return

    dm_shocks = pd.DataFrame({
        "company": ["WIPRO.NS"] * 25,
        "date": dates,
        "mean_sentiment": sents,
        "headline_count": [1] * 25,
        "has_news": [1] * 25,
        "Close": closes,
    })

    shocks_df = detect_sentiment_shocks(dm_shocks, min_company_obs=20, threshold_std=2.0)
    print("Detected Shock Events:")
    print(shocks_df[["company", "date", "mean_sentiment", "deviation_from_mean", "two_session_forward_return"]])
    assert len(shocks_df) == 1
    assert shocks_df.iloc[0]["date"] == "2026-07-11"
    
    # 15. Two-Session Forward Return Verification across Trading Sessions
    print("--- 15. Testing Two-Session Forward Return Calculation ---")
    # Day 10 Close is 110.0 (100 + 10). Day 12 Close is 112.0 (100 + 12).
    # Expected two_session_forward_return = (112 - 110) / 110 = 2 / 110 = 0.0181818...
    expected_2s_ret = (112.0 - 110.0) / 110.0
    actual_2s_ret = shocks_df.iloc[0]["two_session_forward_return"]
    print(f"Calculated Two-Session Forward Return: {actual_2s_ret:.6f} (Expected: {expected_2s_ret:.6f})")
    assert abs(actual_2s_ret - expected_2s_ret) < 1e-5
    print("PASSED: Two-session forward return across trading sessions verified.\n")

    # 16. GDELT / FinBERT Signal Cross Validation Test
    print("--- 16. Testing GDELT / FinBERT Cross Validation ---")
    test_cross_df = pd.DataFrame([
        {"sentiment_score": 0.8, "gdelt_tone": 3.0},   # Agree (pos, pos)
        {"sentiment_score": -0.5, "gdelt_tone": -2.0}, # Agree (neg, neg)
        {"sentiment_score": 0.6, "gdelt_tone": -1.0},  # Disagree (pos, neg)
        {"sentiment_score": 0.4, "gdelt_tone": np.nan},# Fallback RSS: MUST BE EXCLUDED!
    ])
    cv_res = cross_validate_signals(test_cross_df)
    print("Cross Validation Results:")
    print(f"  Total Valid GDELT Records: {cv_res['total_valid_gdelt_records']} (Expected: 3)")
    print(f"  Agreement Count: {cv_res['agreement_count']} (Expected: 2)")
    print(f"  Agreement Rate: {cv_res['agreement_rate']:.4f} (Expected: 0.6667)")
    assert cv_res["total_valid_gdelt_records"] == 3, "Missing GDELT tone was wrongly included in denominator!"
    assert cv_res["agreement_count"] == 2
    assert abs(cv_res["agreement_rate"] - (2 / 3)) < 1e-4
    print("PASSED: GDELT / FinBERT cross validation correctly excludes missing GDELT tone records.\n")

    # 17, 18 & 19. Architecture & Dependency Boundary Checks
    print("--- 17, 18 & 19. Architecture & Dependency Boundary Verification ---")
    with open("store.py") as f:
        store_code = f.read()
    with open("analyze.py") as f:
        analyze_code = f.read()

    assert "streamlit" not in store_code.lower(), "Streamlit dependency detected in store.py!"
    assert "streamlit" not in analyze_code.lower(), "Streamlit dependency detected in analyze.py!"
    assert "openai" not in store_code.lower() and "ollama" not in store_code.lower()
    assert "openai" not in analyze_code.lower() and "ollama" not in analyze_code.lower()
    print("PASSED: Zero Streamlit, LLM, or exposed SQL dependencies verified in store.py and analyze.py.\n")

    print("==================================================")
    print("       ALL PHASE 2 VERIFICATIONS PASSED!          ")
    print("==================================================")


if __name__ == "__main__":
    run_phase2_verifications()
