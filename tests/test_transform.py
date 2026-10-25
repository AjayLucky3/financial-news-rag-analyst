"""
Unit tests for transform.py data transformation functions.
Verifies sentiment score bounds [-1, 1], trading session alignment, next-session returns, and NaN preservation.
"""

import os
import sys
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transform import (
    FinBERTSentimentAnalyzer,
    map_timestamp_to_trading_session,
    align_news_to_trading_sessions,
    calculate_returns,
)


def test_sentiment_score_bounds():
    """Test FinBERT sentiment score strictly bounded within [-1.0, +1.0]."""
    analyzer = FinBERTSentimentAnalyzer(batch_size=2)
    headlines = [
        "Company achieves record quarterly revenue and profit expansion.",
        "Severe accounting fraud investigation leads to bankruptcy declaration."
    ]
    results = analyzer.analyze_texts(headlines)

    assert len(results) == 2
    for r in results:
        score = r["sentiment_score"]
        assert -1.0 <= score <= 1.0
        assert 0.0 <= r["prob_positive"] <= 1.0
        assert 0.0 <= r["prob_negative"] <= 1.0
        assert 0.0 <= r["prob_neutral"] <= 1.0


def test_weekend_trading_session_mapping():
    """Test Saturday headline timestamp maps to next Monday trading session."""
    saturday_ts = pd.Timestamp("2026-09-19 15:30:00")
    trading_dates = pd.DatetimeIndex([
        pd.Timestamp("2026-09-18"),
        pd.Timestamp("2026-09-21"),
        pd.Timestamp("2026-09-22"),
    ])

    mapped_date = map_timestamp_to_trading_session(saturday_ts, trading_dates)
    assert mapped_date == pd.Timestamp("2026-09-21")


def test_no_news_nan_preservation():
    """Test days without news items remain NaN (not imputed with zero)."""
    trading_dates = pd.DatetimeIndex([
        pd.Timestamp("2026-09-21"),
        pd.Timestamp("2026-09-22"),
    ])

    mock_news_df = pd.DataFrame([
        {
            "timestamp": pd.Timestamp("2026-09-21 09:00:00"),
            "title": "Positive Monday News",
            "sentiment_score": 0.75,
        }
    ])

    mock_price_df = pd.DataFrame(
        {"Close": [100.0, 102.0]},
        index=trading_dates
    )

    aligned = align_news_to_trading_sessions(mock_news_df, mock_price_df)

    # 2026-09-21 has news
    assert aligned.loc[pd.Timestamp("2026-09-21"), "news_count"] == 1
    assert abs(aligned.loc[pd.Timestamp("2026-09-21"), "sentiment_score"] - 0.75) < 1e-4

    # 2026-09-22 has NO news -> MUST BE NaN
    assert aligned.loc[pd.Timestamp("2026-09-22"), "news_count"] == 0
    assert np.isnan(aligned.loc[pd.Timestamp("2026-09-22"), "sentiment_score"])


def test_calculate_returns_trading_sessions():
    """Test next-session return calculation based on trading date index."""
    trading_dates = pd.DatetimeIndex([
        pd.Timestamp("2026-09-18"), # Friday
        pd.Timestamp("2026-09-21"), # Monday
        pd.Timestamp("2026-09-22"), # Tuesday
    ])

    mock_price = pd.DataFrame({"Close": [100.0, 105.0, 102.9]}, index=trading_dates)
    ret_df = calculate_returns(mock_price)

    # On Friday (100.0), next trading session is Monday (105.0) -> return = 0.05
    assert abs(ret_df.loc[pd.Timestamp("2026-09-18"), "next_day_return"] - 0.05) < 1e-4
    # Last date has no next session return -> NaN
    assert np.isnan(ret_df.loc[pd.Timestamp("2026-09-22"), "next_day_return"])
