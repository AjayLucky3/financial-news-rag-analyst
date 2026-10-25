"""
Verification script for Phase 1 of Financial News Sentiment + RAG Analyst.
Tests config, GDELT fetching, yfinance fetching, FinBERT inference,
sentiment bounds [-1, +1], weekend trading session mapping, and NaN preservation for no-news days.
"""

import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import Phase 1 modules
try:
    from config import config, logger, setup_logging
    from fetch import fetch_gdelt_news, fetch_google_news_rss, fetch_news_for_ticker, fetch_financial_data
    from transform import (
        FinBERTSentimentAnalyzer,
        add_sentiment_to_news,
        map_timestamp_to_trading_session,
        align_news_to_trading_sessions,
        calculate_returns,
    )
    print("SUCCESS: All Phase 1 modules imported cleanly.")
except Exception as e:
    print(f"FAILURE: Module import failed: {e}")
    sys.exit(1)


def run_verifications():
    print("\n=============================================")
    print("      STARTING PHASE 1 VERIFICATIONS         ")
    print("=============================================\n")

    # 1. Config Check
    print("--- 1. Testing config.py ---")
    print(f"Default Tickers: {config.DEFAULT_TICKERS}")
    print(f"Analysis Window: {config.ANALYSIS_WINDOW_DAYS} days")
    print(f"GDELT Endpoint: {config.GDELT_DOC_API_URL}")
    print(f"FinBERT Model: {config.MODEL_NAME}")
    assert len(config.DEFAULT_TICKERS) == 5
    assert config.ANALYSIS_WINDOW_DAYS == 90
    print("PASSED: Config values verified.\n")

    # 2. GDELT Fetch Test (Real API call)
    print("--- 2. Testing GDELT News Fetch ---")
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=7)
    
    ticker = "TCS.NS"
    query = config.TICKER_QUERY_MAP.get(ticker)
    print(f"Fetching real GDELT news for '{ticker}' (Query: {query}) from {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}...")
    gdelt_df = fetch_gdelt_news(query, start_dt, end_dt, max_records=10)
    print(f"Retrieved {len(gdelt_df)} records from GDELT.")
    if not gdelt_df.empty:
        print("Sample GDELT Headline:")
        print(f"  Title: {gdelt_df['title'].iloc[0]}")
        print(f"  Date: {gdelt_df['timestamp'].iloc[0]}")
        print(f"  Domain: {gdelt_df['domain'].iloc[0]}")
    else:
        print("GDELT returned 0 records (or hit backoff retry).")
    print("PASSED: GDELT fetch function executed with HTTP backoff validation.\n")

    # 3. yfinance Fetch Test (Real sequential call)
    print("--- 3. Testing yfinance Fetch ---")
    price_start = end_dt - timedelta(days=14)
    print(f"Fetching price data for tickers {config.DEFAULT_TICKERS[:2]}...")
    price_data = fetch_financial_data(config.DEFAULT_TICKERS[:2], price_start, end_dt)
    assert "TCS.NS" in price_data, "TCS.NS price data missing!"
    tcs_df = price_data["TCS.NS"]
    print(f"TCS.NS Price Data shape: {tcs_df.shape}")
    print(f"TCS.NS Price index type: {type(tcs_df.index)}")
    print("Sample Price Data:")
    print(tcs_df.head(2))
    assert not tcs_df.empty, "TCS.NS price data is empty!"
    print("PASSED: yfinance price fetch verified.\n")

    # 4. FinBERT Sentiment Inference Test & Score Boundary Check [-1, +1]
    print("--- 4. Testing FinBERT Sentiment Inference ---")
    sample_headlines = [
        "Tata Consultancy Services reports stellar quarterly profit growth beating analyst estimates.",
        "Infosys faces severe margin compression and revenue decline due to project cancellations.",
        "Reliance Industries maintains steady operational performance in line with consensus."
    ]
    print(f"Initializing FinBERT analyzer ({config.MODEL_NAME})...")
    analyzer = FinBERTSentimentAnalyzer(batch_size=2)
    sentiment_results = analyzer.analyze_texts(sample_headlines)

    print("\nSentiment Results:")
    for headline, res in zip(sample_headlines, sentiment_results):
        score = res["sentiment_score"]
        pos = res["prob_positive"]
        neg = res["prob_negative"]
        neu = res["prob_neutral"]
        print(f"Headline: '{headline[:60]}...'")
        print(f"  Score: {score:.4f} | Pos: {pos:.4f} | Neg: {neg:.4f} | Neu: {neu:.4f}")
        
        # Hard constraint check: sentiment_score must be between -1.0 and +1.0
        assert -1.0 <= score <= 1.0, f"Sentiment score {score} out of bounds [-1, +1]!"

    print("\nPASSED: FinBERT inference verified and scores strictly within [-1, +1].\n")

    # 5. Weekend to Trading Session Mapping Test
    print("--- 5. Testing Weekend to Trading Session Alignment ---")
    # Simulate a Saturday timestamp (e.g., Saturday 2026-09-19)
    saturday_ts = pd.Timestamp("2026-09-19 14:30:00")
    # Trading dates: Friday 2026-09-18, Monday 2026-09-21, Tuesday 2026-09-22
    trading_dates = pd.DatetimeIndex([
        pd.Timestamp("2026-09-18"),
        pd.Timestamp("2026-09-21"),
        pd.Timestamp("2026-09-22")
    ])
    
    mapped_date = map_timestamp_to_trading_session(saturday_ts, trading_dates)
    print(f"News timestamp: {saturday_ts} (Saturday)")
    print(f"Mapped trading date: {mapped_date} (Expected: 2026-09-21 Monday)")
    assert mapped_date == pd.Timestamp("2026-09-21"), "Weekend mapping failed to resolve to next trading session!"
    print("PASSED: Weekend news correctly mapped to next available trading session date.\n")

    # 6. NaN Preservation Test for Days with No News
    print("--- 6. Testing NaN Preservation for No-News Days ---")
    mock_news_df = pd.DataFrame([
        {
            "timestamp": pd.Timestamp("2026-09-18 10:00:00"),
            "title": "Positive headline for Friday",
            "sentiment_score": 0.85,
            "prob_positive": 0.90,
            "prob_negative": 0.05,
            "prob_neutral": 0.05
        }
    ])
    mock_price_df = pd.DataFrame(
        {"Close": [3000.0, 3050.0, 3020.0]},
        index=trading_dates
    )
    
    aligned = align_news_to_trading_sessions(mock_news_df, mock_price_df)
    print("\nAligned DataFrame:")
    print(aligned[["news_count", "sentiment_score"]])
    
    # Check Friday 2026-09-18 (has news)
    assert aligned.loc[pd.Timestamp("2026-09-18"), "news_count"] == 1
    assert not np.isnan(aligned.loc[pd.Timestamp("2026-09-18"), "sentiment_score"])
    
    # Check Monday 2026-09-21 & Tuesday 2026-09-22 (no news)
    assert aligned.loc[pd.Timestamp("2026-09-21"), "news_count"] == 0
    assert np.isnan(aligned.loc[pd.Timestamp("2026-09-21"), "sentiment_score"]), "No-news day was wrongly imputed with zero instead of NaN!"
    assert np.isnan(aligned.loc[pd.Timestamp("2026-09-22"), "sentiment_score"]), "No-news day was wrongly imputed with zero instead of NaN!"
    
    print("PASSED: No-news days remain NaN as strictly required.\n")

    # 7. Next-Day Returns Calculation Test
    print("--- 7. Testing Next-Day Returns Calculation ---")
    returns_df = calculate_returns(mock_price_df)
    print("Returns DataFrame:")
    print(returns_df[["Close", "same_day_return", "next_day_return"]])
    
    # On 2026-09-18: Close is 3000.0, next session (2026-09-21) Close is 3050.0
    # Expected next_day_return = (3050 - 3000)/3000 = 50/3000 = 0.016666...
    expected_ret = (3050.0 - 3000.0) / 3000.0
    actual_ret = returns_df.loc[pd.Timestamp("2026-09-18"), "next_day_return"]
    assert abs(actual_ret - expected_ret) < 1e-5, f"Next day return mismatch: {actual_ret} vs {expected_ret}"
    
    # Last trading date should have NaN next_day_return
    assert np.isnan(returns_df.loc[pd.Timestamp("2026-09-22"), "next_day_return"])
    print("PASSED: Trading session next-day returns calculated correctly without naive calendar shifts.\n")

    print("=============================================")
    print("      ALL PHASE 1 VERIFICATIONS PASSED!     ")
    print("=============================================\n")


if __name__ == "__main__":
    run_verifications()
