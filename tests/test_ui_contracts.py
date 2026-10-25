"""
Unit tests for beginner UX and UI contracts in Financial News Sentiment + RAG Analyst.
Verifies dynamic company sample questions, sentiment interpretation bands,
statistical association plain-English translations, data sufficiency tiers,
and empty-data fallback guarantees.
"""

import os
import sys
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import config, get_display_name
from analyze import (
    assess_data_sufficiency,
    get_sentiment_band,
    get_association_interpretation,
    detect_sentiment_shocks,
)
from rag import get_sample_questions_for_company, answer_query


def test_dynamic_sample_questions_no_cross_company_pollution():
    """Verify that sample questions dynamically adapt to target company without cross-pollution."""
    tcs_questions = get_sample_questions_for_company("TCS.NS")
    infy_questions = get_sample_questions_for_company("INFY.NS")

    assert len(tcs_questions) == 4
    assert len(infy_questions) == 4

    # TCS questions must reference TCS and NEVER reference Infosys or Reliance
    for q in tcs_questions:
        assert "TCS" in q["label"] or "Tata" in q["label"]
        assert "Infosys" not in q["query"]
        assert "Reliance" not in q["query"]

    # INFY questions must reference Infosys and NEVER reference TCS
    for q in infy_questions:
        assert "Infosys" in q["label"] or "INFY" in q["label"]
        assert "TCS" not in q["query"]


def test_sentiment_interpretation_bands():
    """Verify the 5 distinct UI sentiment interpretation bands."""
    strongly_neg = get_sentiment_band(-0.75)
    assert strongly_neg["label"] == "Strongly Negative"
    assert "🔴" in strongly_neg["badge"]

    neg = get_sentiment_band(-0.35)
    assert neg["label"] == "Negative"
    assert "🟠" in neg["badge"]

    mixed = get_sentiment_band(0.05)
    assert mixed["label"] == "Near Neutral / Mixed"
    assert "⚪" in mixed["badge"]

    pos = get_sentiment_band(0.40)
    assert pos["label"] == "Positive"
    assert "🟢" in pos["badge"]

    strongly_pos = get_sentiment_band(0.85)
    assert strongly_pos["label"] == "Strongly Positive"
    assert "🟢" in strongly_pos["badge"]


def test_association_interpretation_tiers():
    """Verify plain-English association translations across sample size and r values."""
    # N < 15 must suppress correlation
    suppressed = get_association_interpretation(r=0.65, valid_obs=10)
    assert "Suppressed" in suppressed["value_display"]
    assert "🔴" in suppressed["badge"]

    # N >= 15 with positive r
    pos_assoc = get_association_interpretation(r=0.45, valid_obs=20)
    assert pos_assoc["label"] == "Positive Association"
    assert "higher sentiment tended to occur alongside higher subsequent returns" in pos_assoc["explanation"].lower()

    # N >= 15 with negative r
    neg_assoc = get_association_interpretation(r=-0.35, valid_obs=22)
    assert neg_assoc["label"] == "Negative Association"
    assert "lower subsequent returns" in neg_assoc["explanation"].lower()

    # N >= 15 with near zero r
    weak_assoc = get_association_interpretation(r=0.08, valid_obs=25)
    assert weak_assoc["label"] == "Very Weak Association"
    assert "little linear association" in weak_assoc["explanation"].lower()


def test_data_sufficiency_tiers():
    """Verify 3-tier data sufficiency labels and badges."""
    low = assess_data_sufficiency(8)
    assert low["tier"] == "INSUFFICIENT DATA"
    assert low["allow_correlation"] is False
    assert "🔴" in low["badge"]

    med = assess_data_sufficiency(18)
    assert med["tier"] == "EXPLORATORY"
    assert med["allow_correlation"] is True
    assert "🟡" in med["badge"]

    high = assess_data_sufficiency(45)
    assert high["tier"] == "BROADER SAMPLE"
    assert high["allow_correlation"] is True
    assert "🟢" in high["badge"]


def test_sentiment_shock_empty_state_safety():
    """Verify shock detection handles empty or sub-threshold company data safely."""
    df_empty = pd.DataFrame()
    with pytest.raises(ValueError):
        detect_sentiment_shocks(df_empty, min_company_obs=20)

    df_small = pd.DataFrame({
        "company": ["TCS.NS"] * 12,
        "date": [f"2026-09-{i+1:02d}" for i in range(12)],
        "mean_sentiment": [0.1] * 12,
        "has_news": [1] * 12,
    })
    shocks_small = detect_sentiment_shocks(df_small, min_company_obs=20)
    assert shocks_small.empty
