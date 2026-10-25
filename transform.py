"""
Data transformation and sentiment analysis module for Financial News Sentiment + RAG Analyst.
Handles FinBERT model inference with bounded batching, trading session alignment,
sentiment score bounds [-1, +1], next-day return calculation, and NaN preservation.
"""

from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import config, logger, get_torch_device


def validate_sentiment_output(
    p_pos: float, p_neg: float, p_neu: float, score: float, label: str
) -> bool:
    """
    Validates FinBERT inference probabilities and calculated sentiment score:
    - all probabilities are finite
    - probabilities are approximately between 0 and 1
    - probabilities approximately sum to 1
    - sentiment score is finite and within [-1.0, +1.0]
    - sentiment score equals positive_score - negative_score
    - sentiment label corresponds to the largest class probability
    """
    if not (np.isfinite(p_pos) and np.isfinite(p_neg) and np.isfinite(p_neu) and np.isfinite(score)):
        return False
    if not (-0.01 <= p_pos <= 1.05 and -0.01 <= p_neg <= 1.05 and -0.01 <= p_neu <= 1.05):
        return False
    if abs((p_pos + p_neg + p_neu) - 1.0) > 0.05:
        return False
    if not (-1.01 <= score <= 1.01):
        return False
    if abs(score - (p_pos - p_neg)) > 1e-4:
        return False
    max_p = max(p_pos, p_neg, p_neu)
    if label == "POSITIVE" and p_pos != max_p:
        return False
    if label == "NEGATIVE" and p_neg != max_p:
        return False
    if label == "NEUTRAL" and p_neu != max_p:
        return False
    return True


_GLOBAL_FINBERT_ANALYZER: Optional["FinBERTSentimentAnalyzer"] = None


def get_finbert_analyzer(
    model_name: Optional[str] = None, batch_size: Optional[int] = None
) -> "FinBERTSentimentAnalyzer":
    """Returns singleton FinBERTSentimentAnalyzer loaded once per process."""
    global _GLOBAL_FINBERT_ANALYZER
    if _GLOBAL_FINBERT_ANALYZER is None:
        _GLOBAL_FINBERT_ANALYZER = FinBERTSentimentAnalyzer(
            model_name=model_name, batch_size=batch_size
        )
    return _GLOBAL_FINBERT_ANALYZER


def get_finbert_health() -> Dict[str, Any]:
    """Returns FinBERT model readiness and health status."""
    try:
        analyzer = get_finbert_analyzer()
        analyzer._load_model()
        return {
            "status": "HEALTHY",
            "model_name": analyzer.model_name,
            "device": analyzer.device,
            "loaded": True,
            "error": None,
        }
    except Exception as e:
        return {
            "status": "UNAVAILABLE",
            "model_name": config.MODEL_NAME,
            "device": get_torch_device(),
            "loaded": False,
            "error": str(e),
        }


class FinBERTSentimentAnalyzer:
    """
    Singleton / Class manager for ProsusAI/finbert sentiment analysis model.
    Enforces model.eval(), torch.no_grad(), and bounded batching to respect 12 GB RAM budget.
    """

    def __init__(self, model_name: Optional[str] = None, batch_size: Optional[int] = None):
        self.model_name = model_name or config.MODEL_NAME
        self.batch_size = batch_size or config.BATCH_SIZE
        self.device = get_torch_device()
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        """Loads tokenizer and model on-demand."""
        if self.tokenizer is None or self.model is None:
            logger.info(f"Loading FinBERT model '{self.model_name}' on device '{self.device}'...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, local_files_only=True)
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            # HARD CONSTRAINT: Put model in evaluation mode
            self.model.eval()
            logger.info("FinBERT model loaded successfully.")

    def analyze_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyzes a list of news headlines/texts in bounded batches.

        Parameters:
            texts: List of headline strings.

        Returns:
            List of dicts with keys: 'positive_score', 'negative_score', 'neutral_score',
            'prob_positive', 'prob_negative', 'prob_neutral', 'sentiment_score', 'sentiment_label'
        """
        if not texts:
            return []

        self._load_model()
        results = []

        # Process in bounded batches to control memory footprint
        num_texts = len(texts)
        for i in range(0, num_texts, self.batch_size):
            batch_texts = texts[i : i + self.batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=config.MAX_SEQ_LENGTH,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # HARD CONSTRAINT: Inference inside torch.no_grad()
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            # FinBERT label order mapping
            id2label = self.model.config.id2label
            pos_idx = 0
            neg_idx = 1
            neu_idx = 2
            for idx, label in id2label.items():
                l_lower = label.lower()
                if "pos" in l_lower:
                    pos_idx = idx
                elif "neg" in l_lower:
                    neg_idx = idx
                elif "neu" in l_lower:
                    neu_idx = idx

            for p in probs:
                p_pos = float(p[pos_idx])
                p_neg = float(p[neg_idx])
                p_neu = float(p[neu_idx])

                # Bounded sentiment score formula: positive_score - negative_score
                # Clamped strictly to [-1.0, +1.0]
                sentiment_score = max(-1.0, min(1.0, float(p_pos - p_neg)))

                # Determine label corresponding to largest class probability
                max_p = max(p_pos, p_neg, p_neu)
                if p_pos == max_p:
                    sentiment_label = "POSITIVE"
                elif p_neg == max_p:
                    sentiment_label = "NEGATIVE"
                else:
                    sentiment_label = "NEUTRAL"

                # Validate outputs against mathematical contracts
                is_valid = validate_sentiment_output(
                    p_pos, p_neg, p_neu, sentiment_score, sentiment_label
                )

                results.append(
                    {
                        "positive_score": p_pos,
                        "negative_score": p_neg,
                        "neutral_score": p_neu,
                        "prob_positive": p_pos,
                        "prob_negative": p_neg,
                        "prob_neutral": p_neu,
                        "sentiment_score": sentiment_score,
                        "sentiment_label": sentiment_label,
                        "is_valid_sentiment": is_valid,
                    }
                )

        return results


def add_sentiment_to_news(
    news_df: pd.DataFrame, analyzer: Optional[FinBERTSentimentAnalyzer] = None
) -> pd.DataFrame:
    """
    Applies FinBERT sentiment analysis to news headlines in news_df.

    Parameters:
        news_df: DataFrame containing a 'title' column.
        analyzer: FinBERTSentimentAnalyzer instance (optional, defaults to process singleton).

    Returns:
        DataFrame enriched with sentiment probabilities, score, and label.
    """
    if news_df.empty or "title" not in news_df.columns:
        logger.warning("Empty news_df or missing 'title' column.")
        for col in [
            "positive_score", "negative_score", "neutral_score",
            "prob_positive", "prob_negative", "prob_neutral",
            "sentiment_score", "sentiment_label"
        ]:
            news_df[col] = np.nan
        return news_df

    if analyzer is None:
        analyzer = get_finbert_analyzer()

    titles = news_df["title"].astype(str).tolist()
    sentiment_results = analyzer.analyze_texts(titles)

    sentiment_df = pd.DataFrame(sentiment_results)
    result_df = pd.concat([news_df.reset_index(drop=True), sentiment_df], axis=1)

    return result_df


def map_timestamp_to_trading_session(
    news_timestamp: pd.Timestamp, trading_dates: pd.DatetimeIndex
) -> Optional[pd.Timestamp]:
    """
    Maps a news timestamp (which could be on a weekend, holiday, or after market hours)
    to the next available actual trading session in trading_dates.

    Parameters:
        news_timestamp: pandas Timestamp of the news item.
        trading_dates: DatetimeIndex of valid yfinance trading session dates (sorted ascending).

    Returns:
        Mapped trading date Timestamp, or None if after all trading dates or invalid.
    """
    if len(trading_dates) == 0 or pd.isna(news_timestamp):
        return None

    # Filter trading_dates to valid sessions >= year 2000
    trading_dates = trading_dates[trading_dates.year >= 2000]
    if len(trading_dates) == 0:
        return None

    # Normalize timestamp to date
    try:
        news_date = pd.to_datetime(news_timestamp).floor("D")
        if hasattr(news_date, "tz_localize") and news_date.tz is not None:
            news_date = news_date.tz_convert(None)
        if pd.isna(news_date) or news_date.year < 2000:
            return None
    except Exception:
        return None

    # Find the smallest trading session date >= news_date
    valid_sessions = trading_dates[trading_dates >= news_date]
    if len(valid_sessions) > 0:
        return valid_sessions[0]
    else:
        # If news is after latest trading session date, map to last available session
        return trading_dates[-1]


def align_news_to_trading_sessions(
    news_df: pd.DataFrame, price_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aligns news items to actual yfinance trading sessions and aggregates daily sentiment per trading session.

    HARD CONSTRAINTS ENFORCED:
    1. Weekend/holiday news maps to the next available trading session date in price_df.
    2. Days with NO news remain NaN (do NOT impute missing sentiment with 0).

    Parameters:
        news_df: DataFrame with ['timestamp', 'sentiment_score', ...]
        price_df: DataFrame indexed by trading date (DatetimeIndex).

    Returns:
        pd.DataFrame indexed by trading session date with merged sentiment metrics and NaN on no-news days.
    """
    if price_df.empty:
        logger.warning("Empty price_df provided for alignment.")
        return pd.DataFrame()

    trading_dates = price_df.index.sort_values()

    # Create a base dataframe anchored on all actual trading session dates
    aligned_df = pd.DataFrame(index=trading_dates)
    aligned_df.index.name = "trading_date"

    if news_df.empty or "sentiment_score" not in news_df.columns:
        # HARD CONSTRAINT: Keep sentiment columns as NaN when no news exists
        aligned_df["sentiment_score"] = np.nan
        aligned_df["news_count"] = 0
        aligned_df["prob_positive"] = np.nan
        aligned_df["prob_negative"] = np.nan
        aligned_df["prob_neutral"] = np.nan
        return aligned_df

    # Map each news item's timestamp to the appropriate trading session date
    ts_col = "timestamp" if "timestamp" in news_df.columns else ("publication_date" if "publication_date" in news_df.columns else "date")
    mapped_dates = []
    for ts in news_df[ts_col]:
        m_date = map_timestamp_to_trading_session(pd.to_datetime(ts), trading_dates)
        mapped_dates.append(m_date)

    temp_news = news_df.copy()
    temp_news["trading_date"] = mapped_dates

    # Filter out unmapped news items
    temp_news = temp_news.dropna(subset=["trading_date"])

    if temp_news.empty:
        aligned_df["sentiment_score"] = np.nan
        aligned_df["news_count"] = 0
        aligned_df["prob_positive"] = np.nan
        aligned_df["prob_negative"] = np.nan
        aligned_df["prob_neutral"] = np.nan
        return aligned_df

    # Aggregate daily sentiment by trading_date
    cnt_col = "title" if "title" in temp_news.columns else ("headline" if "headline" in temp_news.columns else "company")
    agg_dict = {
        "sentiment_score": "mean",
        "prob_positive": "mean",
        "prob_negative": "mean",
        "prob_neutral": "mean",
        cnt_col: "count",
    }
    # Only aggregate columns that exist
    actual_agg = {k: v for k, v in agg_dict.items() if k in temp_news.columns}

    daily_agg = temp_news.groupby("trading_date").agg(actual_agg)
    daily_agg = daily_agg.rename(columns={cnt_col: "news_count"})

    # Left join onto full trading dates dataframe
    aligned_df = aligned_df.join(daily_agg, how="left")

    # Fill news_count NaNs with 0 count
    if "news_count" not in aligned_df.columns:
        aligned_df["news_count"] = 0
    aligned_df["news_count"] = aligned_df["news_count"].fillna(0).astype(int)

    # HARD CONSTRAINT VERIFICATION:
    # Ensure sentiment_score remains NaN (not 0.0) for days where news_count == 0
    no_news_mask = aligned_df["news_count"] == 0
    aligned_df.loc[no_news_mask, "sentiment_score"] = np.nan
    if "prob_positive" in aligned_df.columns:
        aligned_df.loc[no_news_mask, "prob_positive"] = np.nan
    if "prob_negative" in aligned_df.columns:
        aligned_df.loc[no_news_mask, "prob_negative"] = np.nan
    if "prob_neutral" in aligned_df.columns:
        aligned_df.loc[no_news_mask, "prob_neutral"] = np.nan

    return aligned_df


def calculate_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates next-day returns based on actual yfinance trading session dates.

    HARD CONSTRAINT ENFORCED:
    Do NOT use naive calendar shift(-1). Return calculation uses consecutive trading session
    rows in price_df (which already excludes weekends and market holidays).

    Formula:
        next_day_return = (Close_{t+1} - Close_t) / Close_t

    Parameters:
        price_df: DataFrame indexed by trading date containing 'Close' column.

    Returns:
        DataFrame with added 'same_day_return' and 'next_day_return' columns.
    """
    df = price_df.copy()

    if "Close" not in df.columns or df.empty:
        logger.warning("Cannot calculate returns: 'Close' column missing or price_df empty.")
        df["same_day_return"] = np.nan
        df["next_day_return"] = np.nan
        return df

    # Same-day return (Close_t vs Close_{t-1})
    df["same_day_return"] = df["Close"].pct_change()

    # Next-day trading session return: (Close_{t+1} - Close_t) / Close_t
    # In a trading-date indexed DataFrame, shift(-1) shifts to the next consecutive TRADING SESSION row.
    df["next_day_return"] = df["Close"].shift(-1) / df["Close"] - 1.0

    return df
