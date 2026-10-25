"""
Final Phase 1 Verification Script.
Checks Hugging Face cache for ProsusAI/finbert, performs controlled GDELT fetch,
verifies yfinance market data, and validates FinBERT inference.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

from config import config, logger
from fetch import fetch_gdelt_news_raw, fetch_gdelt_news, fetch_financial_data
from transform import FinBERTSentimentAnalyzer


def verify_hf_cache():
    print("--- 1 & 2. Hugging Face Cache Check ---")
    hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    print(f"HuggingFace Hub Cache Path: {hf_cache_dir}")
    if not hf_cache_dir.exists():
        print("HF Cache directory does not exist yet.")
        return False, "Not Found"
    
    # Look for model directories
    model_dirs = list(hf_cache_dir.glob("models--*finbert*"))
    if model_dirs:
        print(f"Found cached model repo(s): {[m.name for m in model_dirs]}")
        exact_id = model_dirs[0].name.replace("models--", "").replace("--", "/")
        print(f"Exact Cached Repository/Model Identifier: {exact_id}")
        return True, exact_id
    else:
        print("No finbert model folder found in HuggingFace cache.")
        return False, "Not Found"


def verify_controlled_gdelt():
    print("\n--- 3. Controlled GDELT API Request ---")
    company = "Tata Consultancy Services"
    ticker = "TCS.NS"
    query = f'"{company}" OR "TCS"'
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=7)
    start_str = start_dt.strftime("%Y%m%d%H%M%S")
    end_str = end_dt.strftime("%Y%m%d%H%M%S")

    print(f"Querying GDELT for '{company}' ({start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')})...")
    
    # Raw HTTP call test for exact metadata recording
    import requests
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "startdatetime": start_str,
        "enddatetime": end_str,
        "maxrecords": "10",
    }
    
    res = requests.get(config.GDELT_DOC_API_URL, params=params, timeout=15)
    http_status = res.status_code
    content_type = res.headers.get("Content-Type", "")
    print(f"HTTP Status: {http_status}")
    print(f"Content-Type: {content_type}")

    is_json = False
    num_records = 0
    sample_headline = None
    pub_timestamp = None
    gdelt_tone_extracted = False

    if http_status == 200 and "json" in content_type.lower():
        try:
            data = res.json()
            is_json = True
            articles = data.get("articles", [])
            num_records = len(articles)
            print(f"JSON returned: Yes | Records count: {num_records}")
            
            if articles:
                sample_headline = articles[0].get("title")
                pub_timestamp = articles[0].get("seendate")
                # GDELT artlist mode does not include tone unless specified or present
                gdelt_tone_extracted = "tone" in articles[0]
                print(f"Sample Headline: {sample_headline}")
                print(f"Publication Timestamp: {pub_timestamp}")
                print(f"GDELT Tone Extracted: {gdelt_tone_extracted}")
        except Exception as e:
            print(f"JSON Parse Exception: {e}")
    else:
        print(f"Response not standard 200 JSON (Status: {http_status}). Body snippet: {res.text[:150]}")

    # Process through fetch_gdelt_news wrapper
    df = fetch_gdelt_news(query, start_dt, end_dt, max_records=10)
    print(f"fetch_gdelt_news returned DataFrame shape: {df.shape}")
    
    return {
        "http_status": http_status,
        "content_type": content_type,
        "is_json": is_json,
        "num_records": num_records,
        "sample_headline": sample_headline,
        "pub_timestamp": pub_timestamp,
        "gdelt_tone_extracted": gdelt_tone_extracted,
    }


def verify_yfinance():
    print("\n--- 4. yfinance Market Data Verification ---")
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=10)
    data = fetch_financial_data(["TCS.NS"], start_dt, end_dt)
    if "TCS.NS" in data and not data["TCS.NS"].empty:
        df = data["TCS.NS"]
        print(f"Retrieved {len(df)} trading sessions for TCS.NS.")
        print(f"Latest Close Price: {df['Close'].iloc[-1]} on {df.index[-1].strftime('%Y-%m-%d')}")
        return True
    else:
        print("Failed to retrieve yfinance data for TCS.NS.")
        return False


def verify_finbert():
    print("\n--- 5. FinBERT Inference Verification ---")
    analyzer = FinBERTSentimentAnalyzer(batch_size=2)
    headlines = ["TCS quarterly profits rise by 15 percent year-over-year."]
    res = analyzer.analyze_texts(headlines)
    if res and len(res) == 1:
        score = res[0]["sentiment_score"]
        print(f"Sample Headline: '{headlines[0]}'")
        print(f"FinBERT Sentiment Score: {score:.4f}")
        assert -1.0 <= score <= 1.0
        return True
    return False


if __name__ == "__main__":
    print("==================================================")
    print("       FINAL PHASE 1 VERIFICATION CHECKS          ")
    print("==================================================")
    
    cached, model_id = verify_hf_cache()
    gdelt_info = verify_controlled_gdelt()
    yf_ok = verify_yfinance()
    finbert_ok = verify_finbert()

    print("\n==================================================")
    print("              VERIFICATION SUMMARY                ")
    print("==================================================")
    print(f"1 & 2. HF FinBERT Cache: {cached} (ID: {model_id})")
    print(f"3. GDELT Status: HTTP {gdelt_info['http_status']} | Content-Type: {gdelt_info['content_type']} | JSON: {gdelt_info['is_json']} | Records: {gdelt_info['num_records']}")
    print(f"4. yfinance Data Active: {yf_ok}")
    print(f"5. FinBERT Inference Valid: {finbert_ok}")
    print("==================================================")
