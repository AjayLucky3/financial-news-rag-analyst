"""
Storage module for Financial News Sentiment + RAG Analyst.
Provides a dual-layer storage system:
1. SQLite (data/finance.db) - Structured numerical source of truth.
2. ChromaDB (data/chroma_db) - Textual and vector retrieval layer using sentence-transformers.
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import config, logger

# Default Storage Paths
DB_DIR: Path = Path(__file__).parent / "data"
SQLITE_DB_PATH: Path = DB_DIR / "finance.db"
CHROMA_DB_PATH: Path = DB_DIR / "chroma_db"
CHROMA_COLLECTION_NAME: str = "financial_news"

# Storage mode configuration state (REAL vs DEMO)
_IS_DEMO_MODE: bool = False
_EMBEDDING_FUNCTION = None


def get_chroma_embedding_function():
    """Returns cached SentenceTransformerEmbeddingFunction for ChromaDB."""
    global _EMBEDDING_FUNCTION
    if _EMBEDDING_FUNCTION is None:
        _EMBEDDING_FUNCTION = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    return _EMBEDDING_FUNCTION


def set_storage_mode(demo: bool = False) -> None:
    """Configures process-level storage mode (REAL vs DEMO)."""
    global _IS_DEMO_MODE
    _IS_DEMO_MODE = demo
    logger.info(f"Storage mode set to: {'DEMO' if demo else 'REAL'}")


def get_storage_mode() -> str:
    """Returns active storage mode string ('REAL' or 'DEMO')."""
    return "DEMO" if _IS_DEMO_MODE else "REAL"


def get_sqlite_path() -> Path:
    """Returns active SQLite database path based on storage mode."""
    if _IS_DEMO_MODE:
        return DB_DIR / "demo_finance.db"
    return SQLITE_DB_PATH


def get_chroma_path() -> Path:
    """Returns active ChromaDB directory path based on storage mode."""
    if _IS_DEMO_MODE:
        return DB_DIR / "demo_chroma_db"
    return CHROMA_DB_PATH


def get_chroma_collection_name() -> str:
    """Returns active ChromaDB collection name based on storage mode."""
    if _IS_DEMO_MODE:
        return "financial_news_demo"
    return "financial_news"


def _get_sqlite_connection() -> sqlite3.Connection:
    """Returns a SQLite connection object for active storage mode with WAL mode enabled."""
    db_path = get_sqlite_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def initialize_database() -> None:
    """
    Initializes SQLite tables ('headlines', 'daily_metrics') and indexes,
    and initializes ChromaDB collection if not present.
    """
    logger.info(f"Initializing SQLite database at '{SQLITE_DB_PATH}'...")
    with _get_sqlite_connection() as conn:
        cursor = conn.cursor()

        # Create 'headlines' table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS headlines (
                article_id TEXT PRIMARY KEY,
                company TEXT NOT NULL,
                ticker TEXT,
                publication_date TEXT NOT NULL,
                headline TEXT NOT NULL,
                cleaned_headline TEXT,
                url TEXT,
                source_type TEXT NOT NULL CHECK(source_type IN ('gdelt', 'fallback')),
                gdelt_tone REAL,
                positive_score REAL,
                negative_score REAL,
                neutral_score REAL,
                sentiment_score REAL,
                sentiment_label TEXT,
                ingestion_timestamp TEXT,
                pipeline_run_id TEXT
            );
            """
        )

        # Create indexes for 'headlines'
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_headlines_company ON headlines(company);")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_headlines_pubdate ON headlines(publication_date);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_headlines_company_pubdate ON headlines(company, publication_date);"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_headlines_article_id ON headlines(article_id);")

        # Create 'daily_metrics' table with explicit market data schema
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_metrics (
                company TEXT NOT NULL,
                ticker TEXT,
                date TEXT NOT NULL,
                close REAL,
                open REAL,
                high REAL,
                low REAL,
                volume REAL,
                mean_sentiment REAL,
                headline_count INTEGER NOT NULL,
                has_news INTEGER NOT NULL,
                daily_return REAL,
                next_session_return REAL,
                valid_sentiment INTEGER,
                valid_return INTEGER,
                rolling_7_observation_corr REAL,
                rolling_14_observation_corr REAL,
                PRIMARY KEY (company, date)
            );
            """
        )

        # Dynamic schema migration for existing databases missing new columns
        existing_cols = [r[1] for r in cursor.execute("PRAGMA table_info(daily_metrics);").fetchall()]
        for col, col_type in [
            ("close", "REAL"),
            ("open", "REAL"),
            ("high", "REAL"),
            ("low", "REAL"),
            ("volume", "REAL"),
            ("rolling_7_observation_corr", "REAL"),
            ("rolling_14_observation_corr", "REAL"),
        ]:
            if col not in existing_cols:
                cursor.execute(f"ALTER TABLE daily_metrics ADD COLUMN {col} {col_type};")

        existing_h_cols = [r[1] for r in cursor.execute("PRAGMA table_info(headlines);").fetchall()]
        if "processing_status" not in existing_h_cols:
            cursor.execute("ALTER TABLE headlines ADD COLUMN processing_status TEXT DEFAULT 'INDEXED_IN_CHROMA';")

        conn.commit()

    logger.info("SQLite database initialization complete.")


def generate_article_id(company: str, headline: str, publication_date: str) -> str:
    """Generates a deterministic SHA256 article_id string."""
    raw_str = f"{company.strip().upper()}_{headline.strip()}_{str(publication_date).strip()}"
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()


def save_headlines(headlines_df: pd.DataFrame) -> None:
    """
    Saves or updates headline records in the SQLite database idempotently.

    Parameters:
        headlines_df: DataFrame containing headline data.
    """
    if headlines_df.empty:
        logger.info("No headlines to save.")
        return

    initialize_database()

    df = headlines_df.copy()

    # Ensure required fields exist
    if "company" not in df.columns:
        df["company"] = df.get("ticker", "UNKNOWN")
    if "source_type" not in df.columns:
        df["source_type"] = df["source"].apply(
            lambda s: "gdelt" if str(s).upper() == "GDELT" else "fallback"
        ) if "source" in df.columns else "fallback"

    # Restrict source_type to allowed set
    df["source_type"] = df["source_type"].apply(
        lambda s: "gdelt" if str(s).lower() in ["gdelt", "gdelt_doc"] else "fallback"
    )

    # Standardize publication_date format
    if "timestamp" in df.columns:
        df["publication_date"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    elif "publication_date" in df.columns:
        df["publication_date"] = pd.to_datetime(df["publication_date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        df["publication_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Generate article_id if missing
    if "article_id" not in df.columns or df["article_id"].isnull().any():
        df["article_id"] = [
            generate_article_id(c, h, p)
            for c, h, p in zip(df["company"], df["title" if "title" in df.columns else "headline"], df["publication_date"])
        ]

    # Map column names to table schema
    if "title" in df.columns and "headline" not in df.columns:
        df["headline"] = df["title"]

    ingestion_ts = datetime.now().isoformat()
    if "ingestion_timestamp" not in df.columns:
        df["ingestion_timestamp"] = ingestion_ts
    if "pipeline_run_id" not in df.columns:
        df["pipeline_run_id"] = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Determine sentiment label if not present
    if "sentiment_label" not in df.columns and "sentiment_score" in df.columns:
        def get_label(score):
            if pd.isna(score):
                return "NEUTRAL"
            if score > 0.15:
                return "POSITIVE"
            elif score < -0.15:
                return "NEGATIVE"
            return "NEUTRAL"
        df["sentiment_label"] = df["sentiment_score"].apply(get_label)

    records_to_insert = []
    for _, row in df.iterrows():
        records_to_insert.append(
            (
                str(row["article_id"]),
                str(row["company"]),
                str(row.get("ticker", "")),
                str(row["publication_date"]),
                str(row["headline"]),
                str(row.get("cleaned_headline", row["headline"])),
                str(row.get("url", "")),
                str(row["source_type"]),
                float(row["gdelt_tone"]) if pd.notna(row.get("gdelt_tone")) else None,
                float(row["prob_positive"]) if pd.notna(row.get("prob_positive")) else None,
                float(row["prob_negative"]) if pd.notna(row.get("prob_negative")) else None,
                float(row["prob_neutral"]) if pd.notna(row.get("prob_neutral")) else None,
                float(row["sentiment_score"]) if pd.notna(row.get("sentiment_score")) else None,
                str(row.get("sentiment_label", "NEUTRAL")),
                str(row["ingestion_timestamp"]),
                str(row["pipeline_run_id"]),
            )
        )

    upsert_query = """
    INSERT OR REPLACE INTO headlines (
        article_id, company, ticker, publication_date, headline, cleaned_headline,
        url, source_type, gdelt_tone, positive_score, negative_score, neutral_score,
        sentiment_score, sentiment_label, ingestion_timestamp, pipeline_run_id
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    with _get_sqlite_connection() as conn:
        cursor = conn.cursor()
        cursor.executemany(upsert_query, records_to_insert)
        conn.commit()

    logger.info(f"Saved {len(records_to_insert)} headline records to SQLite.")


def save_daily_metrics(daily_metrics_df: pd.DataFrame) -> None:
    """
    Saves daily metrics to SQLite database idempotently.
    HARD CONSTRAINT: Does NOT impute missing news sentiment with zero.
    Days with no news keep mean_sentiment = NULL, headline_count = 0, has_news = 0.

    Parameters:
        daily_metrics_df: DataFrame with metrics per company and date.
    """
    if daily_metrics_df.empty:
        logger.info("No daily metrics to save.")
        return

    initialize_database()

    df = daily_metrics_df.copy()

    # Reset index if date/trading_date is in index or column
    if "date" not in df.columns and "trading_date" in df.columns:
        df = df.rename(columns={"trading_date": "date"})
    elif "date" not in df.columns and df.index.name in ["date", "trading_date"]:
        idx_name = df.index.name
        df = df.reset_index()
        df = df.rename(columns={idx_name: "date"})
    elif "date" not in df.columns:
        df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    invalid_mask = df["date"].isna() | (df["date"].dt.year < 2000)
    if invalid_mask.any():
        df.loc[invalid_mask, "date"] = pd.Timestamp.now()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    records_to_insert = []
    for _, row in df.iterrows():
        company = str(row["company"])
        ticker = str(row.get("ticker", ""))
        date_str = str(row["date"])

        # Market price data columns
        close_val = (
            float(row["close"]) if pd.notna(row.get("close")) else (
                float(row["Close"]) if pd.notna(row.get("Close")) else None
            )
        )
        open_val = (
            float(row["open"]) if pd.notna(row.get("open")) else (
                float(row["Open"]) if pd.notna(row.get("Open")) else None
            )
        )
        high_val = (
            float(row["high"]) if pd.notna(row.get("high")) else (
                float(row["High"]) if pd.notna(row.get("High")) else None
            )
        )
        low_val = (
            float(row["low"]) if pd.notna(row.get("low")) else (
                float(row["Low"]) if pd.notna(row.get("Low")) else None
            )
        )
        volume_val = (
            float(row["volume"]) if pd.notna(row.get("volume")) else (
                float(row["Volume"]) if pd.notna(row.get("Volume")) else None
            )
        )

        h_cnt = row.get("headline_count", row.get("news_count", 0))
        headline_count = int(h_cnt) if pd.notna(h_cnt) else 0
        has_news = int(row.get("has_news")) if pd.notna(row.get("has_news")) else (1 if headline_count > 0 else 0)

        # Mean sentiment: MUST remain None / NULL if no news or NaN
        mean_sentiment = float(row["sentiment_score"]) if (has_news and pd.notna(row.get("sentiment_score"))) else (
            float(row["mean_sentiment"]) if (has_news and pd.notna(row.get("mean_sentiment"))) else None
        )
        
        daily_return = float(row["same_day_return"]) if pd.notna(row.get("same_day_return")) else (
            float(row["daily_return"]) if pd.notna(row.get("daily_return")) else None
        )
        next_session_return = float(row["next_day_return"]) if pd.notna(row.get("next_day_return")) else (
            float(row["next_session_return"]) if pd.notna(row.get("next_session_return")) else None
        )

        r7_val = float(row["rolling_7_observation_corr"]) if pd.notna(row.get("rolling_7_observation_corr")) else None
        r14_val = float(row["rolling_14_observation_corr"]) if pd.notna(row.get("rolling_14_observation_corr")) else None

        valid_sentiment = 1 if mean_sentiment is not None else 0
        valid_return = 1 if next_session_return is not None else 0

        records_to_insert.append(
            (
                company,
                ticker,
                date_str,
                close_val,
                open_val,
                high_val,
                low_val,
                volume_val,
                mean_sentiment,
                headline_count,
                has_news,
                daily_return,
                next_session_return,
                valid_sentiment,
                valid_return,
                r7_val,
                r14_val,
            )
        )

    upsert_query = """
    INSERT OR REPLACE INTO daily_metrics (
        company, ticker, date, close, open, high, low, volume,
        mean_sentiment, headline_count, has_news, daily_return, next_session_return,
        valid_sentiment, valid_return, rolling_7_observation_corr, rolling_14_observation_corr
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    with _get_sqlite_connection() as conn:
        cursor = conn.cursor()
        cursor.executemany(upsert_query, records_to_insert)
        conn.commit()

    logger.info(f"Saved {len(records_to_insert)} daily metrics records to SQLite.")


def get_vector_collection() -> chromadb.Collection:
    """
    Returns the ChromaDB Collection instance for vector retrieval.
    Configured with SentenceTransformerEmbeddingFunction for cached local inference.
    """
    DB_DIR.mkdir(parents=True, exist_ok=True)
    chroma_path = get_chroma_path()
    collection_name = get_chroma_collection_name()
    client = chromadb.PersistentClient(path=str(chroma_path))
    ef = get_chroma_embedding_function()

    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"}
        )
    except ValueError as e:
        if "Embedding function conflict" in str(e):
            logger.info("Resolving Chroma collection embedding function conflict...")
            client.delete_collection(name=collection_name)
            collection = client.get_or_create_collection(
                name=collection_name,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"}
            )
        else:
            raise e
    return collection



def save_to_chroma(headlines_df: pd.DataFrame, batch_size: int = 32) -> None:
    """
    Stores headlines in ChromaDB using bounded batching.

    Parameters:
        headlines_df: DataFrame containing headlines and metadata.
        batch_size: Batch size for Chroma insertion.
    """
    if headlines_df.empty:
        logger.info("No headlines to save to Chroma.")
        return

    collection = get_vector_collection()

    df = headlines_df.copy()
    if "company" not in df.columns:
        df["company"] = df.get("ticker", "UNKNOWN")
    if "headline" not in df.columns and "title" in df.columns:
        df["headline"] = df["title"]

    if "source_type" not in df.columns:
        if "source" in df.columns:
            df["source_type"] = df["source"].apply(
                lambda s: "gdelt" if str(s).lower() in ["gdelt", "gdelt_doc"] else "fallback"
            )
        else:
            df["source_type"] = "fallback"
    else:
        df["source_type"] = df["source_type"].apply(
            lambda s: "gdelt" if str(s).lower() in ["gdelt", "gdelt_doc"] else "fallback"
        )

    if "publication_date" not in df.columns and "timestamp" in df.columns:
        df["publication_date"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")
    elif "publication_date" in df.columns:
        df["publication_date"] = pd.to_datetime(df["publication_date"]).dt.strftime("%Y-%m-%d")
    else:
        df["publication_date"] = datetime.now().strftime("%Y-%m-%d")

    if "article_id" not in df.columns or df["article_id"].isnull().any():
        df["article_id"] = [
            generate_article_id(c, h, p)
            for c, h, p in zip(df["company"], df["headline"], df["publication_date"])
        ]

    ids = []
    documents = []
    metadatas = []

    for _, row in df.iterrows():
        art_id = str(row["article_id"])
        text = str(row["headline"])
        pub_date_str = str(row["publication_date"])

        try:
            dt_obj = pd.to_datetime(pub_date_str)
            if pd.isna(dt_obj) or dt_obj.year < 2000:
                dt_obj = pd.Timestamp.now()
            date_int = int(dt_obj.strftime("%Y%m%d"))
            pub_date_str = dt_obj.strftime("%Y-%m-%d")
        except Exception:
            dt_obj = pd.Timestamp.now()
            date_int = int(dt_obj.strftime("%Y%m%d"))
            pub_date_str = dt_obj.strftime("%Y-%m-%d")

        # Construct scalar metadata for Chroma compatibility
        meta: Dict[str, Any] = {
            "company": str(row["company"]),
            "date": pub_date_str,
            "date_int": date_int,
            "url": str(row.get("url", "")),
            "source_type": str(row.get("source_type", "fallback")),
        }

        if pd.notna(row.get("sentiment_score")):
            meta["sentiment_score"] = float(row["sentiment_score"])
        else:
            meta["sentiment_score"] = 0.0

        if pd.notna(row.get("sentiment_label")):
            meta["sentiment_label"] = str(row["sentiment_label"])
        else:
            meta["sentiment_label"] = "NEUTRAL"

        ids.append(art_id)
        documents.append(text)
        metadatas.append(meta)

    # Process in bounded batches to control memory
    total = len(ids)
    for i in range(0, total, batch_size):
        b_ids = ids[i : i + batch_size]
        b_docs = documents[i : i + batch_size]
        b_meta = metadatas[i : i + batch_size]

        collection.upsert(
            ids=b_ids,
            documents=b_docs,
            metadatas=b_meta,
        )

    logger.info(f"Saved {total} headline documents to ChromaDB collection '{get_chroma_collection_name()}'.")


def save_data(headlines_df: pd.DataFrame, daily_metrics_df: pd.DataFrame) -> None:
    """
    Convenience orchestrator function to save headlines and metrics to SQLite and ChromaDB.
    """
    if not headlines_df.empty:
        save_headlines(headlines_df)
        save_to_chroma(headlines_df)

    if not daily_metrics_df.empty:
        save_daily_metrics(daily_metrics_df)


def reindex_chroma_from_sqlite(batch_size: int = 32) -> Dict[str, Any]:
    """
    Rebuilds or synchronizes ChromaDB vector index from stored SQLite headlines.
    Does NOT refetch articles from the internet. Idempotently upserts all stored records.
    """
    initialize_database()
    with _get_sqlite_connection() as conn:
        headlines_df = pd.read_sql_query("SELECT * FROM headlines ORDER BY publication_date ASC;", conn)

    if headlines_df.empty:
        logger.info("Reindex: No headlines found in SQLite to index.")
        return {"status": "SUCCESS", "reindexed_count": 0}

    save_to_chroma(headlines_df, batch_size=batch_size)

    with _get_sqlite_connection() as conn:
        conn.execute("UPDATE headlines SET processing_status = 'INDEXED_IN_CHROMA';")
        conn.commit()

    logger.info(f"Reindex complete: {len(headlines_df)} headlines indexed into ChromaDB.")
    return {
        "status": "SUCCESS",
        "reindexed_count": len(headlines_df),
        "collection": get_chroma_collection_name(),
    }


# --- Structured Query Helper Functions ---

def get_company_sentiment_summary(company: str) -> dict:
    """
    Returns sentiment summary statistics for a company from SQLite.
    """
    with _get_sqlite_connection() as conn:
        cursor = conn.cursor()
        query = """
        SELECT 
            COUNT(article_id) as total_headlines,
            AVG(sentiment_score) as avg_sentiment,
            SUM(CASE WHEN sentiment_score > 0.15 THEN 1 ELSE 0 END) as pos_count,
            SUM(CASE WHEN sentiment_score < -0.15 THEN 1 ELSE 0 END) as neg_count,
            SUM(CASE WHEN sentiment_score BETWEEN -0.15 AND 0.15 THEN 1 ELSE 0 END) as neu_count
        FROM headlines
        WHERE company = ?;
        """
        cursor.execute(query, (company,))
        row = cursor.fetchone()
        if not row or row[0] == 0:
            return {
                "company": company,
                "total_headlines": 0,
                "avg_sentiment": None,
                "pos_count": 0,
                "neg_count": 0,
                "neu_count": 0,
            }
        return {
            "company": company,
            "total_headlines": row[0],
            "avg_sentiment": float(row[1]) if row[1] is not None else None,
            "pos_count": row[2],
            "neg_count": row[3],
            "neu_count": row[4],
        }


def get_worst_sentiment_period(days: int = 7) -> List[dict]:
    """
    Returns companies/dates with the lowest average sentiment over the specified rolling window.
    """
    with _get_sqlite_connection() as conn:
        cursor = conn.cursor()
        query = """
        SELECT company, date, mean_sentiment, headline_count
        FROM daily_metrics
        WHERE mean_sentiment IS NOT NULL
        ORDER BY mean_sentiment ASC
        LIMIT ?;
        """
        cursor.execute(query, (days,))
        rows = cursor.fetchall()
        return [
            {
                "company": r[0],
                "date": r[1],
                "mean_sentiment": r[2],
                "headline_count": r[3],
            }
            for r in rows
        ]


def get_headline_counts_by_company() -> Dict[str, int]:
    """
    Returns total headline count grouped by company.
    """
    with _get_sqlite_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT company, COUNT(article_id) FROM headlines GROUP BY company;"
        cursor.execute(query)
        rows = cursor.fetchall()
        return {r[0]: r[1] for r in rows}


def get_daily_metrics_records(company: Optional[str] = None) -> pd.DataFrame:
    """
    Fetches daily_metrics table from SQLite as a pandas DataFrame.
    Normalizes column aliases (e.g. Close <-> close) to satisfy visualization contracts.
    """
    with _get_sqlite_connection() as conn:
        if company:
            query = "SELECT * FROM daily_metrics WHERE company = ? ORDER BY date ASC;"
            df = pd.read_sql_query(query, conn, params=(company,))
        else:
            query = "SELECT * FROM daily_metrics ORDER BY company, date ASC;"
            df = pd.read_sql_query(query, conn)

    if not df.empty:
        if "close" in df.columns and "Close" not in df.columns:
            df["Close"] = df["close"]
        elif "Close" in df.columns and "close" not in df.columns:
            df["close"] = df["Close"]

    return df
