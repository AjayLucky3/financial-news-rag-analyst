"""
Pipeline Orchestrator for Financial News Sentiment + RAG Analyst.
Executes the end-to-end ingestion and analysis pipeline:
  FETCH -> TRANSFORM -> STORE -> ANALYZE

Maintains per-company failure isolation, unique pipeline run IDs, structured logging,
and optional demo data seeding.
"""

import sys
import argparse
import uuid
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from config import config, logger
from fetch import fetch_news_for_ticker, fetch_financial_data
from transform import (
    FinBERTSentimentAnalyzer,
    add_sentiment_to_news,
    align_news_to_trading_sessions,
    calculate_returns,
)
import store
from store import (
    initialize_database,
    save_headlines,
    save_daily_metrics,
    save_to_chroma,
    save_data,
    set_storage_mode,
    get_storage_mode,
    get_sqlite_path,
    _get_sqlite_connection,
    reindex_chroma_from_sqlite,
)
from analyze import (
    compute_correlations,
    compute_rolling_correlations,
    detect_sentiment_shocks,
    cross_validate_signals,
    assess_data_sufficiency,
)


def seed_demo_data() -> Dict[str, Any]:
    """
    Seeds synthetic demonstration data for dashboard showcase into demo storage layer.
    HARD CONSTRAINT: Visibly tagged with source_type = 'fallback' and pipeline_run_id = 'DEMO_RUN'.
    """
    logger.info("Seeding synthetic demonstration data into DEMO storage mode...")
    set_storage_mode(demo=True)
    initialize_database()

    np.random.seed(42)
    demo_companies = ["TCS.NS", "INFY.NS", "RELIANCE.NS", "WIPRO.NS", "HDFCBANK.NS"]
    start_dt = datetime.now() - timedelta(days=90)
    dates = [start_dt + timedelta(days=i) for i in range(90)]
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]

    headlines_list = []
    metrics_list = []

    for comp in demo_companies:
        for idx, d_str in enumerate(date_strs):
            # 70% of days have news
            if np.random.rand() > 0.3:
                score = float(np.random.uniform(-0.9, 0.9))
                label = "POSITIVE" if score > 0.15 else ("NEGATIVE" if score < -0.15 else "NEUTRAL")
                title = f"[DEMO DATA] {comp} operational update and quarterly market report for {d_str}"
                
                headlines_list.append(
                    {
                        "company": comp,
                        "ticker": comp,
                        "timestamp": f"{d_str} 10:00:00",
                        "publication_date": f"{d_str} 10:00:00",
                        "title": title,
                        "headline": title,
                        "url": f"https://demo.example.com/{comp.lower().replace('.', '_')}_{idx}",
                        "source": "fallback",
                        "source_type": "fallback",
                        "gdelt_tone": score * 3.0,
                        "prob_positive": max(0.0, score) if score > 0 else 0.1,
                        "prob_negative": abs(min(0.0, score)) if score < 0 else 0.1,
                        "prob_neutral": 0.2,
                        "sentiment_score": score,
                        "sentiment_label": label,
                        "pipeline_run_id": "DEMO_RUN",
                    }
                )
                head_cnt = 1
                mean_sent = score
            else:
                head_cnt = 0
                mean_sent = np.nan  # NaN preservation on no-news days

            ret = float(np.random.normal(0.001, 0.015))
            next_ret = float(np.random.normal(0.001, 0.015))

            c_price = 1000.0 + idx * 5.0 + float(np.random.normal(0, 10))
            metrics_list.append(
                {
                    "company": comp,
                    "ticker": comp,
                    "date": d_str,
                    "mean_sentiment": mean_sent if head_cnt > 0 else np.nan,
                    "sentiment_score": mean_sent if head_cnt > 0 else np.nan,
                    "headline_count": head_cnt,
                    "has_news": 1 if head_cnt > 0 else 0,
                    "daily_return": ret,
                    "same_day_return": ret,
                    "next_session_return": next_ret,
                    "next_day_return": next_ret,
                    "Close": c_price,
                    "close": c_price,
                    "open": c_price - 2.0,
                    "high": c_price + 5.0,
                    "low": c_price - 5.0,
                    "volume": 100000 + idx * 500,
                }
            )

    demo_h_df = pd.DataFrame(headlines_list)
    demo_m_df = pd.DataFrame(metrics_list)

    save_headlines(demo_h_df)
    save_daily_metrics(demo_m_df)
    save_to_chroma(demo_h_df)

    logger.info("Synthetic demo data successfully seeded into SQLite and Chroma (DEMO mode).")
    return {
        "status": "SUCCESS",
        "mode": "DEMO_DATA_SEEDED",
        "demo_headlines_count": len(demo_h_df),
        "demo_metrics_count": len(demo_m_df),
    }


def run_pipeline(
    companies: Optional[List[str]] = None,
    days: Optional[int] = None,
    seed_demo: bool = False,
) -> Dict[str, Any]:
    """
    Executes the end-to-end data fetching, sentiment analysis, storage, and statistical pipeline.

    Parameters:
        companies: List of company tickers (defaults to config.DEFAULT_TICKERS).
        days: Historical window in days (defaults to config.ANALYSIS_WINDOW_DAYS).
        seed_demo: If True, seeds synthetic demonstration data.

    Returns:
        Structured pipeline result dictionary containing run metrics and status.
    """
    if seed_demo:
        return seed_demo_data()

    target_companies = companies or config.DEFAULT_TICKERS
    window_days = days or config.ANALYSIS_WINDOW_DAYS
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

    start_time = time.time()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=window_days)

    logger.info("==================================================")
    logger.info(f"STARTING PIPELINE RUN: {run_id}")
    logger.info(f"Companies: {target_companies} | Window: {window_days} days")
    logger.info("==================================================")

    initialize_database()

    # Step 1: Sequential Financial Data Fetching (yfinance)
    logger.info("--- Stage 1: Fetching yfinance Price Data ---")
    price_data_map = fetch_financial_data(target_companies, start_date, end_date)

    if not price_data_map:
        err_msg = "All price data requests failed. Pipeline cannot proceed without price history."
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    analyzer = FinBERTSentimentAnalyzer()

    all_headlines = []
    all_aligned_metrics = []
    company_errors = {}

    # Step 2: Per-Company Failure-Isolated Fetch & Transform Pipeline
    for comp in target_companies:
        logger.info(f"--- Processing Pipeline for {comp} ---")
        try:
            # Fetch News (GDELT -> Google News RSS Fallback)
            news_df = fetch_news_for_ticker(comp, start_date, end_date)

            if not news_df.empty:
                # Add FinBERT Sentiment
                news_df = add_sentiment_to_news(news_df, analyzer)
                news_df["pipeline_run_id"] = run_id
                news_df["company"] = comp
                save_headlines(news_df)
                save_to_chroma(news_df)
                all_headlines.append(news_df)

            # Query all stored headlines for this company from SQLite for complete session alignment
            with _get_sqlite_connection() as conn:
                comp_news_df = pd.read_sql_query(
                    "SELECT * FROM headlines WHERE company = ? OR ticker = ?",
                    conn,
                    params=(comp, comp),
                )

            # Retrieve price DataFrame for alignment
            price_df = price_data_map.get(comp, pd.DataFrame())
            if not price_df.empty:
                # Calculate returns
                price_df_ret = calculate_returns(price_df)

                # Align news sentiment to trading sessions using all stored headlines
                aligned_df = align_news_to_trading_sessions(comp_news_df, price_df_ret)

                # Assign market price columns while aligned_df and price_df_ret share identical index
                aligned_df["company"] = comp
                aligned_df["ticker"] = comp
                aligned_df["close"] = price_df_ret.get("Close", price_df_ret.get("close", np.nan))
                aligned_df["Close"] = aligned_df["close"]
                aligned_df["open"] = price_df_ret.get("Open", price_df_ret.get("open", np.nan))
                aligned_df["high"] = price_df_ret.get("High", price_df_ret.get("high", np.nan))
                aligned_df["low"] = price_df_ret.get("Low", price_df_ret.get("low", np.nan))
                aligned_df["volume"] = price_df_ret.get("Volume", price_df_ret.get("volume", np.nan))
                aligned_df["same_day_return"] = price_df_ret.get("same_day_return", np.nan)
                aligned_df["next_session_return"] = price_df_ret.get("next_day_return", np.nan)
                aligned_df["daily_return"] = aligned_df["same_day_return"]

                # Ensure headline_count and has_news are properly populated
                h_cnt = aligned_df.get("news_count", aligned_df.get("headline_count", 0))
                aligned_df["headline_count"] = h_cnt.fillna(0).astype(int)
                aligned_df["has_news"] = (aligned_df["headline_count"] > 0).astype(int)
                aligned_df["mean_sentiment"] = aligned_df.get("sentiment_score", aligned_df.get("mean_sentiment", np.nan))

                # Reset index and standardize date column
                aligned_df = aligned_df.reset_index()
                date_col = "trading_date" if "trading_date" in aligned_df.columns else aligned_df.columns[0]
                aligned_df = aligned_df.rename(columns={date_col: "date"})
                aligned_df["date"] = pd.to_datetime(aligned_df["date"]).dt.strftime("%Y-%m-%d")

                all_aligned_metrics.append(aligned_df)

        except Exception as e:
            logger.error(f"Error processing pipeline stage for {comp}: {e}")
            company_errors[comp] = str(e)

    # Step 3: Combine and Store Data
    logger.info("--- Stage 3: Saving Data to SQLite and ChromaDB ---")
    combined_headlines_df = (
        pd.concat(all_headlines, axis=0).reset_index(drop=True)
        if all_headlines
        else pd.DataFrame()
    )
    combined_metrics_df = (
        pd.concat(all_aligned_metrics, axis=0).reset_index(drop=True)
        if all_aligned_metrics
        else pd.DataFrame()
    )

    if not combined_headlines_df.empty:
        save_headlines(combined_headlines_df)
        save_to_chroma(combined_headlines_df)

    if not combined_metrics_df.empty:
        save_daily_metrics(combined_metrics_df)

    # Step 4: Run Statistical Analysis
    logger.info("--- Stage 4: Running Quantitative Statistical Analysis ---")
    analysis_results = {}
    if not combined_metrics_df.empty:
        if "mean_sentiment" not in combined_metrics_df.columns and "sentiment_score" in combined_metrics_df.columns:
            combined_metrics_df["mean_sentiment"] = combined_metrics_df["sentiment_score"]

        # Calculate rolling correlations over valid news observations
        rolling_metrics_df = compute_rolling_correlations(combined_metrics_df)
        save_daily_metrics(rolling_metrics_df)

        corr_res = compute_correlations(rolling_metrics_df, min_obs=15)
        shocks_df = detect_sentiment_shocks(rolling_metrics_df, combined_headlines_df, min_company_obs=20)
        cross_val = cross_validate_signals(combined_headlines_df)

        analysis_results = {
            "correlations": corr_res,
            "shocks_count": len(shocks_df),
            "cross_validation": cross_val,
        }

    duration = time.time() - start_time
    logger.info("==================================================")
    logger.info(f"PIPELINE RUN COMPLETED: {run_id} (Duration: {duration:.2f}s)")
    logger.info("==================================================")

    return {
        "pipeline_run_id": run_id,
        "status": "COMPLETED",
        "duration_seconds": round(duration, 2),
        "companies_processed": [c for c in target_companies if c not in company_errors],
        "company_errors": company_errors,
        "total_headlines_ingested": len(combined_headlines_df),
        "total_metrics_records": len(combined_metrics_df),
        "analysis_results": analysis_results,
    }


def generate_data_quality_report() -> Dict[str, Any]:
    """
    Generates a comprehensive data quality and audit report across all supported companies.
    Validates headline counts, trading sessions, comparable sessions, data sufficiency status,
    price continuity, date integrity, and demo/real storage isolation.
    """
    initialize_database()
    storage_mode = get_storage_mode()
    sqlite_path = store.get_sqlite_path()

    report: Dict[str, Any] = {
        "storage_mode": storage_mode,
        "database_path": str(sqlite_path),
        "companies": {},
        "integrity_checks": {
            "no_legacy_dates": True,
            "unique_company_dates": True,
            "price_continuity_valid": True,
            "demo_isolation_clean": True,
            "anomalies_detected": [],
        },
    }

    with _get_sqlite_connection() as conn:
        # Check demo isolation
        demo_cnt = conn.execute(
            "SELECT COUNT(*) FROM headlines WHERE pipeline_run_id = 'DEMO_RUN'"
        ).fetchone()[0]
        if not store._IS_DEMO_MODE and demo_cnt > 0:
            report["integrity_checks"]["demo_isolation_clean"] = False
            report["integrity_checks"]["anomalies_detected"].append(
                f"Contamination: Found {demo_cnt} DEMO_RUN headlines in REAL storage."
            )

        # Check legacy/bad dates (< 2000-01-01)
        bad_dates = conn.execute(
            "SELECT company, date FROM daily_metrics WHERE date < '2000-01-01'"
        ).fetchall()
        if bad_dates:
            report["integrity_checks"]["no_legacy_dates"] = False
            report["integrity_checks"]["anomalies_detected"].append(
                f"Legacy dates detected in daily_metrics: {bad_dates}"
            )

        all_metrics = pd.read_sql_query("SELECT * FROM daily_metrics ORDER BY date ASC", conn)
        all_headlines = pd.read_sql_query("SELECT * FROM headlines", conn)

        print("\n" + "=" * 80)
        print("  DATA QUALITY & PIPELINE AUDIT REPORT")
        print("=" * 80)
        print(f"Storage Mode : {report['storage_mode']}")
        print(f"Database     : {report['database_path']}")
        print("-" * 80)

        header_fmt = "{:<12} | {:<9} | {:<9} | {:<12} | {:<11} | {:<25}"
        print(header_fmt.format("Company", "Headlines", "News Days", "Trading Sess", "Comparable", "Status"))
        print("-" * 80)

        for ticker in config.DEFAULT_TICKERS:
            h_sub = all_headlines[all_headlines["company"] == ticker] if not all_headlines.empty else pd.DataFrame()
            m_sub = all_metrics[all_metrics["company"] == ticker] if not all_metrics.empty else pd.DataFrame()

            h_count = len(h_sub)
            m_count = len(m_sub)
            news_days = len(m_sub[m_sub["has_news"] == 1]) if not m_sub.empty else 0

            comparable_sessions = 0
            if not m_sub.empty:
                comp_mask = (
                    m_sub["mean_sentiment"].notna()
                    & m_sub["next_session_return"].notna()
                    & (m_sub["has_news"] == 1)
                )
                comparable_sessions = int(comp_mask.sum())

            sufficiency = assess_data_sufficiency(comparable_sessions)

            # Check logical consistency
            if h_count == 0 and news_days > 0:
                report["integrity_checks"]["anomalies_detected"].append(
                    f"{ticker}: Inconsistency: 0 stored headlines but {news_days} news days."
                )

            # Check price continuity
            if not m_sub.empty and "close" in m_sub.columns and m_sub["close"].notna().sum() > 1:
                closes = m_sub["close"].dropna().values
                pct_changes = np.abs(np.diff(closes) / closes[:-1])
                max_change = float(np.max(pct_changes)) if len(pct_changes) > 0 else 0.0
                if max_change > 0.50:
                    report["integrity_checks"]["price_continuity_valid"] = False
                    report["integrity_checks"]["anomalies_detected"].append(
                        f"{ticker}: Sudden price change of {max_change*100:.1f}% detected."
                    )

            report["companies"][ticker] = {
                "headlines": h_count,
                "news_days": news_days,
                "trading_sessions": m_count,
                "comparable_sessions": comparable_sessions,
                "status": sufficiency["badge"],
            }

            print(header_fmt.format(
                ticker, h_count, news_days, m_count, comparable_sessions, sufficiency["badge"]
            ))

        print("-" * 80)
        if report["integrity_checks"]["anomalies_detected"]:
            print("⚠️ ANOMALIES DETECTED:")
            for anom in report["integrity_checks"]["anomalies_detected"]:
                print(f"  - {anom}")
        else:
            print("✅ All integrity checks PASSED (dates valid, prices continuous, demo isolated).")
        print("=" * 80 + "\n")

    return report


def main():
    """CLI Entry point for orchestrator."""
    parser = argparse.ArgumentParser(description="Financial News Sentiment + RAG Pipeline Orchestrator")
    parser.add_argument("--run-pipeline", action="store_true", help="Execute ingestion and analysis pipeline")
    parser.add_argument("--backfill", action="store_true", help="Execute historical backfill (default 365 days)")
    parser.add_argument("--days", type=int, default=None, help="Historical window size in days (default 90, or 365 for backfill)")
    parser.add_argument("--company", type=str, default=None, help="Single target company ticker (e.g. TCS.NS)")
    parser.add_argument("--seed-demo", action="store_true", help="Seed synthetic demonstration data for dashboard")
    parser.add_argument("--data-quality-report", action="store_true", help="Print data quality and pipeline audit report")
    parser.add_argument("--reindex-chroma", action="store_true", help="Reindex all SQLite headlines into ChromaDB vector store")

    args = parser.parse_args()

    if args.seed_demo:
        res = seed_demo_data()
        print(f"Demo Seeding Result: {res}")
        return

    if args.reindex_chroma:
        print("Reindexing SQLite headlines into ChromaDB...")
        res = reindex_chroma_from_sqlite()
        print(f"Reindex Result: {res}")
        return

    if args.data_quality_report:
        generate_data_quality_report()
        return

    if args.backfill:
        backfill_days = args.days or 365
        print(f"Starting {backfill_days}-day historical backfill...")
        companies = [args.company] if args.company else None
        res = run_pipeline(companies=companies, days=backfill_days)
        print("\n--- Backfill Execution Summary ---")
        print(f"Run ID: {res['pipeline_run_id']}")
        print(f"Status: {res['status']}")
        print(f"Duration: {res['duration_seconds']}s")
        print(f"Companies Processed: {res['companies_processed']}")
        print(f"Headlines Ingested: {res['total_headlines_ingested']}")
        print(f"Metrics Records: {res['total_metrics_records']}")
        generate_data_quality_report()
        return

    if args.run_pipeline or args.days or args.company:
        companies = [args.company] if args.company else None
        res = run_pipeline(companies=companies, days=args.days)
        print("\n--- Pipeline Execution Summary ---")
        print(f"Run ID: {res['pipeline_run_id']}")
        print(f"Status: {res['status']}")
        print(f"Duration: {res['duration_seconds']}s")
        print(f"Companies Processed: {res['companies_processed']}")
        print(f"Headlines Ingested: {res['total_headlines_ingested']}")
        print(f"Metrics Records: {res['total_metrics_records']}")
        if res["company_errors"]:
            print(f"Company Errors: {res['company_errors']}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
