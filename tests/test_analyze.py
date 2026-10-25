"""
Unit tests for analyze.py statistical analysis layer.
Verifies Pearson/Spearman correlations, minimum observation thresholds (<15 and <20),
rolling observation correlations, sentiment shock detection, two-session forward returns, and signal cross validation.
"""

import os
import sys
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analyze import (
    compute_correlations,
    compute_rolling_correlations,
    detect_sentiment_shocks,
    cross_validate_signals,
    assess_data_sufficiency,
)


def test_assess_data_sufficiency_tiers():
    """Verify data sufficiency classification tiers (<15, 15-29, >=30)."""
    tier1 = assess_data_sufficiency(10)
    assert tier1["tier"] == "INSUFFICIENT DATA"
    assert tier1["allow_correlation"] is False

    tier2 = assess_data_sufficiency(20)
    assert tier2["tier"] == "EXPLORATORY"
    assert tier2["allow_correlation"] is True

    tier3 = assess_data_sufficiency(35)
    assert tier3["tier"] == "BROADER SAMPLE"
    assert tier3["allow_correlation"] is True


def test_correlations_min_obs_threshold():
    """Verify that fewer than 15 valid paired observations return NaN correlation."""
    dm_small = pd.DataFrame({
        "company": ["TCS.NS"] * 5,
        "date": [f"2026-09-0{i+1}" for i in range(5)],
        "mean_sentiment": [0.1, 0.2, 0.3, 0.4, 0.5],
        "next_session_return": [0.01, 0.02, 0.01, 0.03, 0.02],
        "has_news": [1] * 5,
    })

    res = compute_correlations(dm_small, min_obs=15)
    comp_res = res["per_company"]["TCS.NS"]

    assert comp_res["insufficient_observations"] is True
    assert np.isnan(comp_res["pearson_corr"])
    assert np.isnan(comp_res["spearman_corr"])


def test_correlations_sufficient_obs():
    """Verify Pearson and Spearman calculation with >= 15 valid observations."""
    n_obs = 20
    s_vals = np.linspace(-0.8, 0.8, n_obs)
    r_vals = 0.5 * s_vals  # Perfectly linear relation

    dm = pd.DataFrame({
        "company": ["RELIANCE.NS"] * n_obs,
        "date": [f"2026-08-{i+1:02d}" for i in range(n_obs)],
        "mean_sentiment": s_vals,
        "next_session_return": r_vals,
        "has_news": [1] * n_obs,
        "headline_count": [1] * n_obs,
    })

    res = compute_correlations(dm, min_obs=15)
    comp_res = res["per_company"]["RELIANCE.NS"]

    assert comp_res["insufficient_observations"] is False
    assert abs(comp_res["pearson_corr"] - 1.0) < 1e-4
    assert abs(comp_res["spearman_corr"] - 1.0) < 1e-4


def test_rolling_observation_correlations():
    """Verify 7-obs and 14-obs rolling correlation calculation over valid news days."""
    n_obs = 15
    s_vals = np.linspace(-0.8, 0.8, n_obs)
    r_vals = s_vals.copy()

    dm = pd.DataFrame({
        "company": ["WIPRO.NS"] * n_obs,
        "date": [f"2026-08-{i+1:02d}" for i in range(n_obs)],
        "mean_sentiment": s_vals,
        "next_session_return": r_vals,
        "has_news": [1] * n_obs,
    })

    df_res = compute_rolling_correlations(dm, min_obs_threshold=5)

    assert "rolling_7_observation_corr" in df_res.columns
    assert "rolling_14_observation_corr" in df_res.columns
    # Row 5 (index 4) should be valid
    assert not np.isnan(df_res["rolling_7_observation_corr"].iloc[4])


def test_sentiment_shock_detection_min_obs():
    """Verify shock detection skips company if valid news observations < 20."""
    dm = pd.DataFrame({
        "company": ["HDFCBANK.NS"] * 10,
        "date": [f"2026-08-{i+1:02d}" for i in range(10)],
        "mean_sentiment": [0.1] * 9 + [0.9],
        "has_news": [1] * 10,
    })

    shocks = detect_sentiment_shocks(dm, min_company_obs=20, threshold_std=2.0)
    assert shocks.empty


def test_cross_validate_signals_excludes_missing_gdelt():
    """Verify fallback RSS articles missing gdelt_tone are excluded from agreement denominator."""
    test_df = pd.DataFrame([
        {"sentiment_score": 0.5, "gdelt_tone": 2.0},   # Agree
        {"sentiment_score": -0.5, "gdelt_tone": -1.0}, # Agree
        {"sentiment_score": 0.5, "gdelt_tone": np.nan},# Missing GDELT tone -> Must be excluded!
    ])

    res = cross_validate_signals(test_df)
    assert res["total_valid_gdelt_records"] == 2
    assert res["agreement_count"] == 2
    assert abs(res["agreement_rate"] - 1.0) < 1e-4
