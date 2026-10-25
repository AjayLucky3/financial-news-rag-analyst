"""
Unit tests for orchestrator.py pipeline orchestration layer.
Verifies run ID generation, company failure isolation, all-company failure handling, and demo data seeding.
"""

import os
import sys
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import store
import orchestrator


@pytest.fixture(autouse=True)
def setup_orchestrator_fixture(tmp_path):
    """Fixture for clean database setup."""
    store.set_storage_mode(demo=False)
    store.SQLITE_DB_PATH = tmp_path / "orch_finance.db"
    store.CHROMA_DB_PATH = tmp_path / "orch_chroma"

    store.initialize_database()
    yield
    store.set_storage_mode(demo=False)


def test_demo_data_seeding():
    """Verify demo data seeding runs cleanly and tags records as DEMO DATA."""
    res = orchestrator.seed_demo_data()
    assert res["status"] == "SUCCESS"
    assert res["mode"] == "DEMO_DATA_SEEDED"

    conn = store._get_sqlite_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM headlines WHERE pipeline_run_id='DEMO_RUN';")
    h_cnt = cur.fetchone()[0]
    conn.close()

    assert h_cnt > 0


@patch("orchestrator.fetch_financial_data")
@patch("orchestrator.fetch_news_for_ticker")
def test_company_failure_isolation(mock_fetch_news, mock_fetch_price):
    """Verify failure in processing one company does not prevent processing remaining companies."""
    # Price data succeeds for TCS.NS and INFY.NS
    mock_price = pd.DataFrame(
        {"Close": [100.0, 105.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-09-20"), pd.Timestamp("2026-09-21")])
    )
    mock_fetch_price.return_value = {"TCS.NS": mock_price, "INFY.NS": mock_price}

    # News fetch throws exception for INFY.NS, succeeds for TCS.NS
    def news_side_effect(ticker, start_date, end_date):
        if ticker == "INFY.NS":
            raise RuntimeError("Network error for INFY")
        return pd.DataFrame([
            {
                "timestamp": pd.Timestamp("2026-09-20 10:00:00"),
                "title": "TCS Cloud growth update",
                "url": "https://example.com/tcs",
                "source": "GDELT",
            }
        ])

    mock_fetch_news.side_effect = news_side_effect

    res = orchestrator.run_pipeline(companies=["TCS.NS", "INFY.NS"], days=7)

    assert res["status"] == "COMPLETED"
    assert "TCS.NS" in res["companies_processed"]
    assert "INFY.NS" in res["company_errors"]


@patch("orchestrator.fetch_financial_data")
def test_all_company_failure(mock_fetch_price):
    """Verify pipeline raises RuntimeError when all financial data calls fail."""
    mock_fetch_price.return_value = {}

    with pytest.raises(RuntimeError):
        orchestrator.run_pipeline(companies=["TCS.NS"], days=7)
