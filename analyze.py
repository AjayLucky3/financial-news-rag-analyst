"""
Statistical Analysis module for Financial News Sentiment + RAG Analyst.
Performs quantitative analysis of news sentiment vs stock market returns.

Contains strictly non-causal statistical correlation analysis, rolling observation correlations,
sentiment shock detection with two-session forward returns, and GDELT/FinBERT cross validation.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from scipy import stats

from config import config, logger


def _validate_columns(df: pd.DataFrame, required_cols: List[str], func_name: str) -> None:
    """Validates presence of required columns in a DataFrame."""
    if df.empty:
        raise ValueError(f"[{func_name}] Input DataFrame is empty.")

    if "mean_sentiment" in required_cols and "mean_sentiment" not in df.columns and "sentiment_score" in df.columns:
        df["mean_sentiment"] = df["sentiment_score"]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"[{func_name}] Missing required columns: {missing}")


def assess_data_sufficiency(valid_obs: int) -> Dict[str, Any]:
    """
    Assesses data sufficiency for statistical inference based on paired valid observations.

    Tiers:
    - N < 15: 🔴 INSUFFICIENT DATA - suppress correlation claim / display 'Not enough observations'
    - 15 <= N <= 29: 🟡 EXPLORATORY SAMPLE - display correlation with exploratory notice
    - 30 <= N <= 59: 🟢 BROADER EXPLORATORY SAMPLE - display correlation with broader sample notice
    - N >= 60: 🟢 LARGER HISTORICAL SAMPLE - display correlation with larger historical sample notice
    """
    if valid_obs < 15:
        return {
            "tier": "INSUFFICIENT DATA",
            "status_code": "INSUFFICIENT",
            "badge": "🔴 INSUFFICIENT DATA",
            "color": "red",
            "description": f"Only {valid_obs} comparable trading sessions available (minimum 15 required for statistical correlation). Correlation metrics suppressed.",
            "allow_correlation": False,
        }
    elif 15 <= valid_obs <= 29:
        return {
            "tier": "EXPLORATORY",
            "status_code": "EXPLORATORY",
            "badge": "🟡 EXPLORATORY SAMPLE",
            "color": "gold",
            "description": f"Small sample size of {valid_obs} comparable sessions. Correlation metrics presented for exploratory analysis only.",
            "allow_correlation": True,
        }
    elif 30 <= valid_obs <= 59:
        return {
            "tier": "BROADER SAMPLE",
            "status_code": "BROADER_EXPLORATORY",
            "badge": "🟢 BROADER EXPLORATORY SAMPLE",
            "color": "green",
            "description": f"Broader sample of {valid_obs} comparable sessions available for comparative analysis.",
            "allow_correlation": True,
        }
    else:
        return {
            "tier": "LARGER HISTORICAL SAMPLE",
            "status_code": "LARGER_HISTORICAL",
            "badge": "🟢 LARGER HISTORICAL SAMPLE",
            "color": "green",
            "description": f"Larger historical sample of {valid_obs} comparable sessions available for longitudinal analysis.",
            "allow_correlation": True,
        }


def get_sentiment_band(score: float) -> Dict[str, Any]:
    """
    Returns presentation band for a given sentiment score based on UI interpretation bands:
    - <= -0.60: Strongly Negative
    - -0.60 to -0.20: Negative
    - -0.20 to +0.20: Near Neutral / Mixed
    - +0.20 to +0.60: Positive
    - >= +0.60: Strongly Positive
    """
    if pd.isna(score):
        return {"label": "N/A", "badge": "⚪ N/A", "color": "gray"}
    if score <= -0.60:
        return {"label": "Strongly Negative", "badge": "🔴 Strongly Negative", "color": "red"}
    elif -0.60 < score <= -0.20:
        return {"label": "Negative", "badge": "🟠 Negative", "color": "orange"}
    elif -0.20 < score < 0.20:
        return {"label": "Near Neutral / Mixed", "badge": "⚪ Near Neutral / Mixed", "color": "gray"}
    elif 0.20 <= score < 0.60:
        return {"label": "Positive", "badge": "🟢 Positive", "color": "green"}
    else:
        return {"label": "Strongly Positive", "badge": "🟢 Strongly Positive", "color": "green"}


def get_association_interpretation(r: float, valid_obs: int) -> Dict[str, Any]:
    """
    Returns UI plain-English translation and interpretation for statistical association (Pearson r):
    - If valid_obs < 15: Suppressed / Insufficient data
    - r >= +0.20: Positive Association
    - -0.20 < r < +0.20: Very Weak Association
    - r <= -0.20: Negative Association
    """
    if valid_obs < 15 or pd.isna(r):
        return {
            "label": "Not enough observations",
            "badge": "🔴 INSUFFICIENT DATA",
            "explanation": "Not enough comparable observations to calculate a meaningful association metric. Try a longer historical analysis period or ingest more historical data.",
            "value_display": "Suppressed (<15 obs)",
        }
    if r >= 0.20:
        return {
            "label": "Positive Association",
            "badge": "🟢 Positive Association",
            "explanation": "Higher sentiment tended to occur alongside higher subsequent returns in this sample. In this dataset, more positive news sentiment tended to co-occur with higher subsequent trading returns. Association describes historical co-movement; it does not prove causation or predict future returns.",
            "value_display": f"{r:+.3f}",
        }
    elif r <= -0.20:
        return {
            "label": "Negative Association",
            "badge": "🔴 Negative Association",
            "explanation": "Higher sentiment tended to occur alongside lower subsequent returns in this sample. In this dataset, more positive news sentiment tended to co-occur with lower subsequent trading returns. Association describes historical co-movement; it does not prove causation or predict future returns.",
            "value_display": f"{r:+.3f}",
        }
    else:
        return {
            "label": "Very Weak Association",
            "badge": "⚪ Very Weak Association",
            "explanation": "Sentiment and subsequent returns showed little linear association in this sample. In this dataset, more positive or negative news sentiment showed very little relationship with the following trading-session return. Association describes historical co-movement; it does not prove causation or predict future returns.",
            "value_display": f"{r:+.3f}",
        }


def compute_correlations(
    daily_metrics: pd.DataFrame, min_obs: int = 15
) -> Dict[str, Any]:
    """
    Computes statistical correlations between daily mean news sentiment and next-session stock returns.

    Statistical Safety:
        This analysis reports observed statistical associations and correlations.
        It does NOT imply causation or guaranteed predictive capability.

    Parameters:
        daily_metrics: DataFrame containing columns ['company', 'mean_sentiment', 'next_session_return', 'has_news']
        min_obs: Minimum required paired valid observations per company (default 15).

    Returns:
        Dict mapping company names to their correlation metrics and descriptive statistics.
    """
    _validate_columns(daily_metrics, ["company", "mean_sentiment", "next_session_return"], "compute_correlations")

    df = daily_metrics.copy()

    # Filter strictly to valid news days where both sentiment and next_session_return are valid (notna)
    valid_mask = (
        df["mean_sentiment"].notna()
        & df["next_session_return"].notna()
        & (df.get("has_news", 1) == 1)
    )
    valid_df = df[valid_mask].copy()

    results: Dict[str, Any] = {"per_company": {}, "portfolio_summary": {}}

    companies = df["company"].unique()
    all_valid_sentiments = []
    all_valid_returns = []

    for company in companies:
        comp_df = valid_df[valid_df["company"] == company].sort_values("date") if "date" in valid_df.columns else valid_df[valid_df["company"] == company]
        n_obs = len(comp_df)
        n_news_days = n_obs
        n_headlines = int(comp_df["headline_count"].sum()) if "headline_count" in comp_df.columns else n_obs
        sufficiency = assess_data_sufficiency(n_obs)

        date_range = "N/A"
        if "date" in comp_df.columns and not comp_df.empty:
            date_range = f"{comp_df['date'].min()} to {comp_df['date'].max()}"

        if n_obs < min_obs:
            logger.info(
                f"Company '{company}' has insufficient valid observations ({n_obs} < {min_obs}). "
                "Returning NaN correlation metrics."
            )
            results["per_company"][company] = {
                "company": company,
                "n_observations": n_obs,
                "n_news_days": n_news_days,
                "n_headlines": n_headlines,
                "date_range": date_range,
                "pearson_corr": np.nan,
                "pearson_pvalue": np.nan,
                "spearman_corr": np.nan,
                "spearman_pvalue": np.nan,
                "insufficient_observations": True,
                "sufficiency": sufficiency,
                "status_message": f"Insufficient observations ({n_obs} < required {min_obs})",
            }
            continue

        sentiments = comp_df["mean_sentiment"].values
        returns = comp_df["next_session_return"].values

        # Calculate Pearson correlation
        p_corr, p_val = stats.pearsonr(sentiments, returns)

        # Calculate Spearman rank correlation
        s_corr, s_val = stats.spearmanr(sentiments, returns)

        results["per_company"][company] = {
            "company": company,
            "n_observations": n_obs,
            "n_news_days": n_news_days,
            "n_headlines": n_headlines,
            "date_range": date_range,
            "pearson_corr": float(p_corr),
            "pearson_pvalue": float(p_val),
            "spearman_corr": float(s_corr),
            "spearman_pvalue": float(s_val),
            "insufficient_observations": False,
            "sufficiency": sufficiency,
            "status_message": "Sufficient observations for correlation computation.",
        }

        all_valid_sentiments.extend(sentiments)
        all_valid_returns.extend(returns)

    # Descriptive portfolio-level aggregate (clearly marked as non-predictive descriptive summary)
    if len(all_valid_sentiments) >= min_obs:
        port_p_corr, port_p_val = stats.pearsonr(all_valid_sentiments, all_valid_returns)
        port_s_corr, port_s_val = stats.spearmanr(all_valid_sentiments, all_valid_returns)
        results["portfolio_summary"] = {
            "description": "Descriptive portfolio-level aggregate (descriptive association only, not predictive)",
            "total_observations": len(all_valid_sentiments),
            "pearson_corr": float(port_p_corr),
            "pearson_pvalue": float(port_p_val),
            "spearman_corr": float(port_s_corr),
            "spearman_pvalue": float(port_s_val),
        }

    return results


def compute_rolling_correlations(
    daily_metrics: pd.DataFrame, min_obs_threshold: int = 5
) -> pd.DataFrame:
    """
    Calculates rolling observation-based correlations for valid news days.

    HARD CONSTRAINTS:
    - Rolling windows are based on VALID news observations, NOT calendar days.
    - Fields are explicitly named 'rolling_7_observation_corr' and 'rolling_14_observation_corr'.
    - Minimum threshold of valid observations per window is enforced.

    Parameters:
        daily_metrics: DataFrame containing daily metrics.
        min_obs_threshold: Minimum paired observations required in rolling window (default 5).

    Returns:
        pd.DataFrame with added rolling correlation columns.
    """
    _validate_columns(daily_metrics, ["company", "mean_sentiment", "next_session_return"], "compute_rolling_correlations")

    df = daily_metrics.copy()
    if "date" in df.columns:
        df = df.sort_values(["company", "date"]).reset_index(drop=True)

    result_dfs = []

    for company, group in df.groupby("company"):
        comp_df = group.copy()

        # Filter strictly to valid news + return observations for rolling calculation
        valid_mask = comp_df["mean_sentiment"].notna() & comp_df["next_session_return"].notna()
        valid_indices = comp_df[valid_mask].index

        rolling_7_series = pd.Series(index=comp_df.index, dtype=float)
        rolling_14_series = pd.Series(index=comp_df.index, dtype=float)

        valid_df = comp_df.loc[valid_indices].copy()
        sentiments = valid_df["mean_sentiment"].tolist()
        returns = valid_df["next_session_return"].tolist()
        v_idx_list = list(valid_indices)

        # Helper to compute rolling Pearson correlation over observation window
        def calc_obs_rolling(window_size: int) -> Dict[int, float]:
            res = {}
            for i in range(len(sentiments)):
                start_i = max(0, i - window_size + 1)
                s_window = sentiments[start_i : i + 1]
                r_window = returns[start_i : i + 1]

                if len(s_window) >= min_obs_threshold and len(set(s_window)) > 1 and len(set(r_window)) > 1:
                    corr, _ = stats.pearsonr(s_window, r_window)
                    res[v_idx_list[i]] = float(corr) if not np.isnan(corr) else np.nan
                else:
                    res[v_idx_list[i]] = np.nan
            return res

        r7_map = calc_obs_rolling(7)
        r14_map = calc_obs_rolling(14)

        for idx, val in r7_map.items():
            rolling_7_series.loc[idx] = val
        for idx, val in r14_map.items():
            rolling_14_series.loc[idx] = val

        comp_df["rolling_7_observation_corr"] = rolling_7_series
        comp_df["rolling_14_observation_corr"] = rolling_14_series
        result_dfs.append(comp_df)

    return pd.concat(result_dfs, axis=0).reset_index(drop=True)


def detect_sentiment_shocks(
    daily_metrics: pd.DataFrame,
    headlines_df: Optional[pd.DataFrame] = None,
    min_company_obs: int = 20,
    threshold_std: float = 2.0,
) -> pd.DataFrame:
    """
    Detects sentiment shock events (defined as days where mean sentiment deviates from company
    mean by at least threshold_std standard deviations) and calculates two-session forward return.

    HARD CONSTRAINTS:
    - Requires at least min_company_obs valid observations per company.
    - two_session_forward_return is calculated using actual trading sessions (Close_{t+2} / Close_t - 1.0).

    Parameters:
        daily_metrics: DataFrame with daily metrics indexed or containing trading date.
        headlines_df: Optional DataFrame with individual headlines.
        min_company_obs: Minimum required valid news observations (default 20).
        threshold_std: Standard deviation multiplier threshold (default 2.0).

    Returns:
        pd.DataFrame containing detected shock records.
    """
    _validate_columns(daily_metrics, ["company", "mean_sentiment"], "detect_sentiment_shocks")

    df = daily_metrics.copy()
    if "date" in df.columns:
        df = df.sort_values(["company", "date"]).reset_index(drop=True)

    shocks = []

    for company, comp_df in df.groupby("company"):
        comp_df = comp_df.reset_index(drop=True)

        # 1. Filter valid news observations
        valid_mask = comp_df["mean_sentiment"].notna() & (comp_df.get("has_news", 1) == 1)
        valid_df = comp_df[valid_mask]

        # 2. Check minimum observation requirement
        if len(valid_df) < min_company_obs:
            logger.info(
                f"Company '{company}' has insufficient valid observations for shock detection "
                f"({len(valid_df)} < {min_company_obs}). Skipping shock detection."
            )
            continue

        # 3 & 4. Calculate company mean and standard deviation of sentiment
        comp_mean = float(valid_df["mean_sentiment"].mean())
        comp_std = float(valid_df["mean_sentiment"].std())

        if np.isnan(comp_std) or comp_std == 0:
            continue

        # Iterate over all trading session rows in comp_df
        for i in range(len(comp_df)):
            row = comp_df.iloc[i]
            mean_sent = row["mean_sentiment"]

            if pd.isna(mean_sent):
                continue

            dev = abs(mean_sent - comp_mean)
            if dev >= threshold_std * comp_std:
                # 5. Calculate two-session forward return across actual trading sessions
                # Shock is on row i (trading session T). We want cumulative return over session T+1 and session T+2.
                # If price data or Close is available, (Close_{i+2} / Close_i) - 1.0
                two_session_ret = np.nan
                if "Close" in comp_df.columns and (i + 2) < len(comp_df):
                    close_t = comp_df.iloc[i]["Close"]
                    close_t2 = comp_df.iloc[i + 2]["Close"]
                    if pd.notna(close_t) and pd.notna(close_t2) and close_t > 0:
                        two_session_ret = float((close_t2 - close_t) / close_t)
                elif "next_session_return" in comp_df.columns and (i + 1) < len(comp_df):
                    # Alternatively combine next_session_return at session i and session i+1
                    r1 = comp_df.iloc[i].get("next_session_return")
                    r2 = comp_df.iloc[i + 1].get("next_session_return")
                    if pd.notna(r1) and pd.notna(r2):
                        two_session_ret = float((1.0 + r1) * (1.0 + r2) - 1.0)

                date_val = row.get("date", f"Session_{i}")
                headline_cnt = int(row.get("headline_count", 1))

                shocks.append(
                    {
                        "company": company,
                        "date": date_val,
                        "mean_sentiment": float(mean_sent),
                        "company_mean": comp_mean,
                        "company_std": comp_std,
                        "deviation_from_mean": float(dev),
                        "deviation_in_std": float(dev / comp_std),
                        "headline_count": headline_cnt,
                        "two_session_forward_return": two_session_ret,
                    }
                )

    cols = [
        "company",
        "date",
        "mean_sentiment",
        "company_mean",
        "company_std",
        "deviation_from_mean",
        "deviation_in_std",
        "headline_count",
        "two_session_forward_return",
    ]
    if not shocks:
        return pd.DataFrame(columns=cols)

    return pd.DataFrame(shocks)[cols]


def cross_validate_signals(headlines_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Cross-validates FinBERT directional sentiment against GDELT tone directional sentiment.

    HARD CONSTRAINT:
    Only includes rows where gdelt_tone is valid (notna()).
    Fallback RSS articles (where gdelt_tone = NaN/None) MUST NOT be included in the denominator.

    Parameters:
        headlines_df: DataFrame containing ['sentiment_score', 'gdelt_tone']

    Returns:
        Dict reporting total valid records, agreement count, and agreement rate.
    """
    _validate_columns(headlines_df, ["sentiment_score"], "cross_validate_signals")

    df = headlines_df.copy()

    if "gdelt_tone" not in df.columns:
        logger.warning("No 'gdelt_tone' column present in headlines DataFrame.")
        return {
            "total_valid_gdelt_records": 0,
            "agreement_count": 0,
            "agreement_rate": np.nan,
            "positive_agreement_count": 0,
            "negative_agreement_count": 0,
            "disagreement_count": 0,
            "status": "No gdelt_tone column found.",
        }

    # Filter strictly to records with valid gdelt_tone
    valid_mask = df["gdelt_tone"].notna() & df["sentiment_score"].notna()
    valid_df = df[valid_mask].copy()

    total_valid = len(valid_df)

    if total_valid == 0:
        logger.info("Zero records with valid GDELT tone for cross-validation.")
        return {
            "total_valid_gdelt_records": 0,
            "agreement_count": 0,
            "agreement_rate": np.nan,
            "positive_agreement_count": 0,
            "negative_agreement_count": 0,
            "disagreement_count": 0,
            "status": "Zero valid GDELT tone records.",
        }

    # FinBERT positive signal: sentiment_score > 0
    finbert_pos = valid_df["sentiment_score"] > 0
    # GDELT tone positive signal: gdelt_tone > 0
    gdelt_pos = valid_df["gdelt_tone"] > 0

    agreement_mask = finbert_pos == gdelt_pos
    agreement_count = int(agreement_mask.sum())
    agreement_rate = float(agreement_count / total_valid)

    pos_agree = int((finbert_pos & gdelt_pos).sum())
    neg_agree = int(((~finbert_pos) & (~gdelt_pos)).sum())
    disagree_count = total_valid - agreement_count

    return {
        "total_valid_gdelt_records": total_valid,
        "agreement_count": agreement_count,
        "agreement_rate": agreement_rate,
        "positive_agreement_count": pos_agree,
        "negative_agreement_count": neg_agree,
        "disagreement_count": disagree_count,
        "status": "Success",
    }
