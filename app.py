"""
Streamlit Executive Dashboard for Financial News Sentiment + RAG Analyst.
Provides a 3-tab executive interface:
  Tab 1: Executive Dashboard (Three-Step Mental Model, Plain-English KPIs, Data Quality Banner, Charts, Company Data Availability)
  Tab 2: Sentiment Shock Events (Unusual sentiment detection & forward returns)
  Tab 3: AI Analyst (RAG natural language QA with decoupled retrieval and grounded evidence)
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import streamlit as st
import pandas as pd
import numpy as np

from config import config, logger, get_display_name
import store
from store import (
    initialize_database,
    get_daily_metrics_records,
    get_company_sentiment_summary,
    get_worst_sentiment_period,
    get_headline_counts_by_company,
    set_storage_mode,
    get_storage_mode,
)
from analyze import (
    compute_correlations,
    compute_rolling_correlations,
    detect_sentiment_shocks,
    cross_validate_signals,
    assess_data_sufficiency,
    get_sentiment_band,
    get_association_interpretation,
)
from visualize import (
    plot_price_over_time,
    plot_daily_sentiment_over_time,
    plot_sentiment_and_returns_over_time,
    plot_sentiment_vs_returns_scatter,
    plot_rolling_correlations,
    plot_sentiment_shocks,
)
import rag

# --- Page Configuration ---
st.set_page_config(
    page_title="Financial News Sentiment + RAG Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

import companies


def compute_data_freshness(company_df: pd.DataFrame) -> Dict[str, str]:
    """Calculates data freshness tier (LIVE / FRESH, RECENT BUT NOT LIVE, STALE, UNAVAILABLE)."""
    if company_df.empty or "date" not in company_df.columns:
        return {
            "status": "UNAVAILABLE",
            "badge": "🔴 UNAVAILABLE",
            "description": "No historical trading or news records loaded in local database.",
            "last_date": "N/A",
        }
    try:
        latest_date_str = str(company_df["date"].max())[:10]
        latest_dt = pd.to_datetime(latest_date_str).date()
        today = pd.Timestamp.now().date()
        diff_days = (today - latest_dt).days

        if diff_days <= 3:
            return {
                "status": "LIVE / FRESH",
                "badge": "🟢 LIVE / FRESH",
                "description": f"Data is current up to {latest_date_str}.",
                "last_date": latest_date_str,
            }
        elif diff_days <= 7:
            return {
                "status": "RECENT BUT NOT LIVE",
                "badge": "🟡 RECENT BUT NOT LIVE",
                "description": f"Latest session was {diff_days} days ago ({latest_date_str}).",
                "last_date": latest_date_str,
            }
        else:
            return {
                "status": "STALE",
                "badge": "🟠 STALE",
                "description": f"Latest session is {diff_days} days old ({latest_date_str}).",
                "last_date": latest_date_str,
            }
    except Exception:
        return {
            "status": "UNKNOWN",
            "badge": "⚪ UNKNOWN",
            "description": "Unable to calculate date freshness.",
            "last_date": "N/A",
        }


# --- Sidebar Controls ---
st.sidebar.title("📊 Analytics Controls")

# Storage Data Mode Selection
data_mode_choice = st.sidebar.radio(
    "Data Storage Mode",
    options=["🟢 REAL Data (finance.db)", "🟡 DEMO Data (demo_finance.db)"],
    index=0,
    help="Toggle between live ingested real data and isolated synthetic demo data."
)
is_demo_mode = "DEMO" in data_mode_choice
set_storage_mode(demo=is_demo_mode)

# Initialize active database
try:
    initialize_database()
except Exception as e:
    st.error(f"Database initialization notice: {e}")

# Equity Universe Selection Mode
universe_mode = st.sidebar.radio(
    "Indian Equity Universe",
    options=["⭐ Default Watchlist (Top 5)", "🔍 All Indian Equities (50+)"],
    index=0,
    help="Select the default 5 NSE giants or explore 50+ Indian equities in the security master.",
)

if "Watchlist" in universe_mode:
    available_companies = companies.get_default_companies()
    company_options = {
        f"{c.short_name} ({c.symbol})": c.symbol
        for c in available_companies
    }
else:
    all_sectors = ["All Sectors"] + companies.get_all_sectors()
    chosen_sector = st.sidebar.selectbox("Filter by Sector", options=all_sectors, index=0)
    sec_filter = None if chosen_sector == "All Sectors" else chosen_sector
    search_term = st.sidebar.text_input("Search Company / Symbol", placeholder="e.g. Tata, SBI, Airtel, ITC")
    available_companies = companies.search_companies(search_term, sector=sec_filter)
    if not available_companies:
        st.sidebar.caption("No matching companies found; showing default watchlist.")
        available_companies = companies.get_default_companies()
    company_options = {
        f"{c.short_name} ({c.symbol}) — {c.sector}": c.symbol
        for c in available_companies
    }

selected_label = st.sidebar.selectbox(
    "Select Target Company",
    options=list(company_options.keys()),
    index=0,
    help="Choose an Indian listed company for sentiment and price correlation analysis.",
)
target_company = company_options[selected_label]
selected_display_name = get_display_name(target_company)

# Analysis Window Selector
window_labels = {
    30: "30 Days — Quick exploratory view",
    60: "60 Days — Short historical view",
    90: "90 Days — Standard quarterly view (Default)",
    180: "180 Days — Extended multi-quarter view",
    365: "365 Days — Full year historical view",
}
analysis_window = st.sidebar.selectbox(
    "Analysis Period",
    options=[30, 60, 90, 180, 365],
    index=2, # Default 90 days
    format_func=lambda d: window_labels.get(d, f"{d} Days"),
    help="Filter historical market and news data window in days."
)

if st.sidebar.button("🔍 Analyze", use_container_width=True):
    st.rerun()

# Fetch daily metrics from SQLite for active storage mode
try:
    all_metrics_df = get_daily_metrics_records()
except Exception as e:
    all_metrics_df = pd.DataFrame()

# Filter by analysis window date cutoff if data exists
if not all_metrics_df.empty and "date" in all_metrics_df.columns:
    try:
        max_d_dt = pd.to_datetime(all_metrics_df["date"]).max()
        min_cutoff = (max_d_dt - pd.Timedelta(days=analysis_window)).strftime("%Y-%m-%d")
        all_metrics_df = all_metrics_df[all_metrics_df["date"] >= min_cutoff].copy()
    except Exception as e:
        logger.warning(f"Error filtering analysis window: {e}")

# Check if active dataset contains synthetic demo records
has_demo_records = False
if not all_metrics_df.empty:
    with store._get_sqlite_connection() as conn:
        try:
            demo_check = conn.execute("SELECT COUNT(*) FROM headlines WHERE pipeline_run_id = 'DEMO_RUN'").fetchone()
            if demo_check and demo_check[0] > 0:
                has_demo_records = True
        except Exception:
            pass

# Sidebar Data Coverage & Metadata
st.sidebar.markdown("---")
st.sidebar.subheader("📌 Data Coverage Metadata")

if is_demo_mode or has_demo_records:
    st.sidebar.warning("🟡 DEMONSTRATION DATA")
    st.sidebar.caption("_These records are synthetic and are used only to demonstrate the application._")
else:
    st.sidebar.success("🟢 REAL INGESTED DATA")

st.sidebar.caption(f"**Storage Mode:** `{get_storage_mode()}`")
st.sidebar.caption(f"**Analysis Period:** {analysis_window} Days")

if not all_metrics_df.empty and "date" in all_metrics_df.columns:
    min_d = all_metrics_df["date"].min()
    max_d = all_metrics_df["date"].max()
    st.sidebar.caption(f"**Date Range:** {min_d} to {max_d}")
    st.sidebar.caption(f"**Total Trading Sessions:** {all_metrics_df['date'].nunique()}")
    
    total_headlines_cnt = 0
    try:
        counts_dict = get_headline_counts_by_company()
        total_headlines_cnt = sum(counts_dict.values())
    except Exception:
        pass
    st.sidebar.caption(f"**Total Ingested Headlines:** {total_headlines_cnt}")
else:
    st.sidebar.info("Database is empty.")

if is_demo_mode:
    if st.sidebar.button("⚡ Seed 90-Day Demo Data", use_container_width=True):
        from orchestrator import seed_demo_data
        seed_demo_data()
        st.rerun()
else:
    st.sidebar.caption("Run historical backfill via:\n`python orchestrator.py --backfill --days 365`")

# System Health & Diagnostics in Sidebar
with st.sidebar.expander("🩺 System Health & Diagnostics"):
    try:
        health_info = rag.get_system_health()
        
        # SQLite
        sql_h = health_info.get("sqlite", {})
        if sql_h.get("status") == "HEALTHY":
            st.markdown(f"**SQLite Database:** `🟢 Connected`\n"
                        f"- Headlines: {sql_h.get('headlines_count', 0)}\n"
                        f"- Market Prices: {sql_h.get('market_prices_count', 0)}\n"
                        f"- Daily Metrics: {sql_h.get('daily_metrics_count', 0)}")
        else:
            st.markdown("**SQLite Database:** `🔴 Error`")

        # Embedding Model
        emb_h = health_info.get("embedding_model", {})
        if emb_h.get("status") == "HEALTHY":
            st.markdown(f"**Embedding Model:** `🟢 Ready`\n- `{emb_h.get('model_name')}`")
        else:
            st.markdown("**Embedding Model:** `🟡 Offline/Unavailable`")

        # ChromaDB
        chr_h = health_info.get("chromadb", {})
        if chr_h.get("status") == "HEALTHY":
            st.markdown(f"**ChromaDB Vector Store:** `🟢 Ready`\n- Documents: {chr_h.get('document_count', 0)}")
        else:
            st.markdown("**ChromaDB Vector Store:** `🔴 Error`")

        # FinBERT
        fb_h = health_info.get("finbert", {})
        if fb_h.get("status") == "HEALTHY":
            st.markdown("**FinBERT Sentiment:** `🟢 Cached Locally`")
        else:
            st.markdown("**FinBERT Sentiment:** `🟡 Ingest Mode`")

        # LLM Provider
        llm_h = health_info.get("llm_provider", {})
        mode = llm_h.get("mode", "LLM UNAVAILABLE")
        if mode == "LLM AVAILABLE":
            st.markdown(f"**LLM Service:** `🟢 {mode}`\n- Provider: {llm_h.get('active_provider')}")
        elif mode == "LLM FALLBACK":
            st.markdown(f"**LLM Service:** `🟡 {mode}`\n- Provider: {llm_h.get('active_provider')}")
        else:
            st.markdown(f"**LLM Service:** `⚪ {mode}`\n- Active Mode: Factual Grounded Fallback")
        st.caption(f"_{llm_h.get('explanation', '')}_")

    except Exception as e:
        st.warning("Diagnostics unavailable.")

company_metrics_df = (
    all_metrics_df[all_metrics_df["company"] == target_company].copy()
    if not all_metrics_df.empty and "company" in all_metrics_df.columns
    else pd.DataFrame()
)

# Header
st.title("Understand a Company's News & Stock")
st.caption(
    "Evidence-grounded financial news sentiment analytics and retrieval-augmented generation engine."
)

# --- Navigation Tabs ---
tab1, tab2, tab3 = st.tabs(
    ["🏢 Executive Dashboard", "⚡ Sentiment Shock Events", "🤖 AI Analyst (RAG)"]
)

# ==============================================================================
# TAB 1: EXECUTIVE DASHBOARD
# ==============================================================================
with tab1:
    st.header(f"Executive Summary — {selected_display_name}")

    # Data Freshness Indicator
    freshness = compute_data_freshness(company_metrics_df)
    st.markdown(f"**Data Freshness:** {freshness['badge']} — _{freshness['description']}_")

    # --- Beginner Onboarding Guide ---
    with st.expander("👋 BEGINNER ONBOARDING GUIDE — START HERE"):
        st.markdown(
            """
            ### Welcome to Financial News Sentiment & Stock Analytics
            
            1. **What does this website do?**
               It analyzes how the linguistic tone of financial news headlines relates to subsequent stock market price movements for major Indian listed equities, using AI (FinBERT) and statistical correlation.
            
            2. **What data does it use?**
               - **Financial News:** Authentic news headlines retrieved from the GDELT Project and Google News RSS fallback.
               - **Market Prices:** Historical daily OHLCV trading session data from Yahoo Finance for NSE-listed equities.
               - **Sentiment Model:** ProsusAI/FinBERT, a financial domain transformer model that scores headline tone from -1.0 to +1.0.
            
            3. **What can I ask it?**
               In the **AI Analyst (RAG)** tab, you can ask questions in plain English:
               - Recent tone: *"What was recent sentiment for TCS?"*
               - Negative themes: *"What were the negative news stories for Infosys?"*
               - Numerical aggregates: *"What is the average sentiment for Reliance?"*
               - Multi-company comparisons: *"Compare TCS and Infosys sentiment."*
            
            4. **What do the results mean?**
               - **Sentiment Score:** Ranges from **-1.0** (strongly negative tone) to **+1.0** (strongly positive tone). 0 is neutral. It describes language tone, NOT stock return probability.
               - **Trading Return:** Percentage price change in the session immediately following the news.
               - **Statistical Association:** Shows whether news and prices historically moved in the same direction. *Correlation is NOT causation and does NOT predict the future.*
            
            5. **What should I do if data is missing?**
               If a company shows no records or insufficient data (<15 sessions), click **"📥 Fetch Live Data"** below to ingest real market history, or switch to **🟡 DEMO Data** mode in the sidebar to test features with synthetic demonstration records.
            """
        )

    if company_metrics_df.empty:
        st.warning(
            f"No historical records found for **{selected_display_name}** ({target_company}) in the active storage mode (`{get_storage_mode()}`)."
        )
        if not is_demo_mode:
            st.markdown("You can fetch live market history and financial news for this company right now:")
            if st.button(f"📥 Fetch Live Data for {selected_display_name}", type="primary"):
                with st.spinner(f"Ingesting live market prices and news for {selected_display_name}..."):
                    try:
                        from orchestrator import run_pipeline
                        run_pipeline(companies=[target_company], days=analysis_window)
                        st.success(f"Successfully fetched data for {selected_display_name}! Reloading dashboard...")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to fetch data for {selected_display_name}: {e}")
        else:
            st.info("Switch to 🟢 REAL Data mode or click '⚡ Seed 90-Day Demo Data' in the sidebar.")
    else:
        # Compute rolling correlations and statistical association metrics
        company_metrics_df = compute_rolling_correlations(company_metrics_df)
        corr_results = compute_correlations(company_metrics_df, min_obs=15)
        comp_corr = corr_results["per_company"].get(target_company, {})

        valid_obs = comp_corr.get("n_observations", 0)
        p_corr = comp_corr.get("pearson_corr", np.nan)
        s_corr = comp_corr.get("spearman_corr", np.nan)
        is_insufficient = comp_corr.get("insufficient_observations", True)

        # Assess data sufficiency
        sufficiency = assess_data_sufficiency(valid_obs)

        # --- Data Quality Banner ---
        if sufficiency["tier"] == "INSUFFICIENT DATA":
            st.error(f"**{sufficiency['badge']}**: {sufficiency['description']}")
        elif sufficiency["tier"] == "EXPLORATORY":
            st.warning(f"**{sufficiency['badge']}**: {sufficiency['description']}")
        else:
            st.success(f"**{sufficiency['badge']}**: {sufficiency['description']}")

        summary = get_company_sentiment_summary(target_company)
        avg_sent_val = summary.get("avg_sentiment")

        st.markdown("---")

        # ======================================================================
        # THREE-STEP MENTAL MODEL
        # ======================================================================
        st.subheader("💡 Three-Step Analytics Model")

        step1_col, step2_col, step3_col = st.columns(3)

        # --- STEP 1: WHAT DID THE NEWS SAY? ---
        with step1_col:
            st.markdown("#### 1. WHAT DID THE NEWS SAY?")
            st.caption("_News Sentiment Analysis_")
            
            if avg_sent_val is not None:
                band = get_sentiment_band(avg_sent_val)
                st.metric(
                    label="Average News Sentiment",
                    value=f"{avg_sent_val:+.3f}",
                    delta=band["label"],
                    delta_color="normal" if avg_sent_val >= 0 else "inverse",
                    help="Balance between positive and negative financial language (-1.0 to +1.0)."
                )
                
                # Visual 3-Pole Score Scale
                st.markdown(f"**Tone Band:** {band['badge']}")
                scale_line = "NEGATIVE (-1.0) ─────── NEUTRAL (0.0) ─────── POSITIVE (+1.0)"
                col_pos = max(10, min(56, int(round(33 + avg_sent_val * 23))))
                pointer_line = " " * col_pos + f"▲ [{avg_sent_val:+.2f}]"
                st.code(f"{scale_line}\n{pointer_line}")
                
                st.caption(
                    f"**Sample Size:** {summary.get('total_headlines', 0)} headlines analyzed.\n\n"
                    f"**What does this score mean?** The score ranges from **-1.0** (strongly negative language) "
                    f"to **0.0** (neutral) to **+1.0** (strongly positive language).\n\n"
                    f"⚠️ **Important:** _This score reflects the linguistic tone detected by FinBERT in news headlines. "
                    f"It is **NOT** a prediction that the stock will rise or fall._"
                )
                
                with st.expander("🔬 How is this calculated?"):
                    st.markdown(
                        f"""
                        Each headline is analyzed by FinBERT to output 3 probabilities:
                        - **Positive Probability ($P_{{pos}}$):** `{summary.get('pos_count', 0) / max(1, summary.get('total_headlines', 1)):.2f}`
                        - **Negative Probability ($P_{{neg}}$):** `{summary.get('neg_count', 0) / max(1, summary.get('total_headlines', 1)):.2f}`
                        - **Neutral Probability ($P_{{neu}}$):** `{summary.get('neu_count', 0) / max(1, summary.get('total_headlines', 1)):.2f}`
                        
                        **Formula:** `Sentiment Score = P(positive) - P(negative)`
                        """
                    )
            else:
                st.metric("Average News Sentiment", "N/A")
                st.caption("No valid news headlines scored.")

        # --- STEP 2: WHAT DID THE STOCK DO? ---
        with step2_col:
            st.markdown("#### 2. WHAT DID THE STOCK DO?")
            st.caption("_Subsequent Trading Session Return_")

            # Strictly compute average next-session return for news-associated trading days
            news_ret_df = company_metrics_df[
                (company_metrics_df["has_news"] == 1)
                & (company_metrics_df["mean_sentiment"].notna())
                & (company_metrics_df["next_session_return"].notna())
            ]
            all_market_df = company_metrics_df[company_metrics_df["next_session_return"].notna()]
            avg_next_ret = float(news_ret_df["next_session_return"].mean()) if not news_ret_df.empty else np.nan
            avg_all_ret = float(all_market_df["next_session_return"].mean()) if not all_market_df.empty else np.nan

            if not np.isnan(avg_next_ret):
                ret_str = f"{avg_next_ret * 100:+.2f}%"
                if avg_next_ret > 0.0005:
                    ret_label = "Average Positive Return"
                    ret_badge = "🟢 Positive"
                    delta_color = "normal"
                elif avg_next_ret < -0.0005:
                    ret_label = "Average Negative Return"
                    ret_badge = "🔴 Negative"
                    delta_color = "inverse"
                else:
                    ret_label = "Flat Return"
                    ret_badge = "⚪ Flat / Neutral"
                    delta_color = "off"

                st.metric(
                    label="Mean Next-Session Return (After News)",
                    value=ret_str,
                    delta=f"{avg_next_ret * 100:+.2f}% ({ret_label})",
                    delta_color=delta_color,
                    help="Average stock percentage change during the next trading session immediately following news.",
                )
                st.markdown(f"**Movement Tone:** {ret_badge}")
                st.caption(
                    f"**Calculated Across:** {len(news_ret_df)} news-associated sessions "
                    f"_(matches {valid_obs} comparable sessions in Step 3)_.\n\n"
                    f"_This measures how the stock moved in the session right after headlines were published._\n\n"
                    f"• **Overall Market Baseline:** {avg_all_ret * 100:+.2f}% across all {len(all_market_df)} trading sessions regardless of news."
                )
            else:
                st.metric("Mean Next-Session Return", "N/A")
                st.caption("No price return data available for news sessions.")

        # --- STEP 3: DID THEY MOVE TOGETHER? ---
        with step3_col:
            st.markdown("#### 3. DID THEY MOVE TOGETHER?")
            st.caption("_Statistical Association_")

            assoc_info = get_association_interpretation(p_corr, valid_obs)

            if not np.isnan(p_corr) and not is_insufficient:
                st.metric(
                    label="Co-movement Association",
                    value=assoc_info["value_display"],
                    delta=assoc_info["label"],
                    help="Pearson linear correlation coefficient (r) measuring co-movement strength."
                )
                st.markdown(f"**Interpretation:** {assoc_info['badge']}")
                st.caption(
                    f"**Comparable Sessions ($N$):** {valid_obs}\n\n"
                    f"_{assoc_info['explanation']}_\n\n"
                    f"⚠️ **Important:** _Association describes historical co-movement. It does not prove that news caused the stock movement and does not predict future returns._"
                )

                with st.expander("🛠️ Technical Association Details"):
                    st.write(f"**Comparable Sessions ($N$):** `{valid_obs}`")
                    st.write(f"**Pearson Correlation ($r$):** `{p_corr:+.4f}`")
                    st.write(f"**Spearman Rank Correlation ($\\rho$):** `{s_corr:+.4f}`")
                    st.write(f"**P-Value:** `{comp_corr.get('pearson_pvalue', np.nan):.4f}`")
            else:
                st.metric(
                    label="Co-movement Association",
                    value="Suppressed (<15 obs)",
                    help="Correlation calculation requires at least 15 paired sessions."
                )
                st.markdown(f"**Status:** {assoc_info['badge']}")
                st.caption(
                    f"**Comparable observations ($N$):** {valid_obs} (requires at least 15 paired sessions)\n\n"
                    f"Because this company has fewer than 15 trading days paired with news in the selected window, "
                    f"calculating a correlation would be statistically unreliable. Co-movement metrics are safely withheld to maintain data integrity.\n\n"
                    f"💡 _Try selecting a longer historical analysis period (e.g. 180 or 365 days) or ingest more historical news._"
                )

        st.markdown("---")

        # --- Data Coverage Panel ("HOW MUCH DATA DO WE HAVE?") ---
        st.subheader("📌 HOW MUCH DATA DO WE HAVE?")
        cov_c1, cov_c2 = st.columns(2)

        with cov_c1:
            st.markdown("##### Global Dataset Coverage")
            st.write(f"• **Analysis Period:** {analysis_window} Days")
            tot_h = sum(get_headline_counts_by_company().values()) if not all_metrics_df.empty else 0
            st.write(f"• **Total Dataset Headlines:** **{tot_h} headlines** across all companies")
            st.write(f"• **Total Trading Sessions:** **{all_metrics_df['date'].nunique() if 'date' in all_metrics_df.columns else 0} sessions**")

        with cov_c2:
            st.markdown(f"##### Selected Company Coverage ({selected_display_name})")
            st.write(f"• **Headlines Analyzed:** **{summary.get('total_headlines', 0)} headlines**")
            news_days_cnt = company_metrics_df[company_metrics_df.get("has_news", 1) == 1]["date"].nunique() if "date" in company_metrics_df.columns else 0
            st.write(f"• **News Days:** **{news_days_cnt} days** with collected news")
            st.write(f"• **Comparable Sessions:** **{valid_obs} sessions** with both sentiment and next-session return")
            st.caption("_Comparable sessions are trading days where both news sentiment and a following return were available._")

        st.markdown("---")

        # --- Plain-English Summary & Findings ---
        st.subheader("📝 Plain-English Summary & Findings")
        summary_bullets = []
        if avg_sent_val is not None:
            band_info = get_sentiment_band(avg_sent_val)
            summary_bullets.append(f"• **Overall News Tone**: News sentiment for **{selected_display_name}** during this period was generally **{band_info['label'].lower()}** ({band_info['badge']}).")
        
        summary_bullets.append(f"• **Data Inventory**: The dataset contains **{summary.get('total_headlines', 0)} headlines** spanning **{valid_obs} comparable trading sessions**.")

        if not np.isnan(p_corr) and not is_insufficient:
            if p_corr >= 0.20:
                assoc_text = "a positive statistical association (days with higher sentiment tended to occur alongside higher subsequent returns)"
            elif p_corr <= -0.20:
                assoc_text = "a negative statistical association (days with higher sentiment tended to occur alongside lower subsequent returns)"
            else:
                assoc_text = "very little linear association (news sentiment showed minimal co-movement with subsequent returns)"
            summary_bullets.append(f"• **Observed Co-movement**: Sentiment and subsequent market returns showed **{assoc_text}**.")
        else:
            summary_bullets.append(f"• **Sample Size Notice**: Currently, **{valid_obs} valid paired sessions** are available (minimum 15 required for statistical correlation confidence).")

        summary_bullets.append("• **Methodology Notice**: Observed associations measure historical co-occurrence in this dataset and **do not establish causation** or guarantee future price movements.")

        st.info("\n".join(summary_bullets))

        st.markdown("---")

        # --- Visualizations Grid ---
        st.subheader("📈 Interactive Charts & Visualizations")
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**Stock Price History**")
            st.caption("Closing price across the selected analysis period.")
            st.plotly_chart(
                plot_price_over_time(company_metrics_df, target_company),
                use_container_width=True,
            )
            
            st.markdown("**Daily News Sentiment Timeline**")
            st.caption("Daily mean FinBERT news sentiment score with zero reference line.")
            st.plotly_chart(
                plot_daily_sentiment_over_time(company_metrics_df, target_company),
                use_container_width=True,
            )

        with col_right:
            st.markdown("**Daily Sentiment vs Subsequent Return Scatter**")
            st.caption("Scatter plot comparing daily sentiment vs next-session return with linear trendline.")
            st.plotly_chart(
                plot_sentiment_vs_returns_scatter(company_metrics_df, target_company),
                use_container_width=True,
            )
            
            st.markdown("**Daily Sentiment & Next-Session Return Dual-Axis**")
            st.caption("Dual-axis comparison of daily sentiment with subsequent session returns.")
            st.plotly_chart(
                plot_sentiment_and_returns_over_time(company_metrics_df, target_company),
                use_container_width=True,
            )

        st.markdown("**Rolling Co-movement Correlations**")
        st.caption("Observation-based rolling correlations over 7 and 14 valid news sessions.")
        st.plotly_chart(
            plot_rolling_correlations(company_metrics_df, target_company),
            use_container_width=True,
        )

        st.markdown("---")

        # --- Company Data Availability Table ---
        st.subheader("📊 COMPANY DATA AVAILABILITY")
        st.markdown("Data coverage and statistical sample readiness across monitored Indian market equities:")

        table_tickers = list(config.DEFAULT_TICKERS)
        if target_company not in table_tickers:
            table_tickers.append(target_company)

        comp_rows = []
        for ticker in table_tickers:
            disp = get_display_name(ticker)
            t_df = (
                all_metrics_df[all_metrics_df["company"] == ticker].copy()
                if not all_metrics_df.empty and "company" in all_metrics_df.columns
                else pd.DataFrame()
            )
            if not t_df.empty:
                t_df = compute_rolling_correlations(t_df)
                t_corr_res = compute_correlations(t_df, min_obs=15)
                t_corr = t_corr_res["per_company"].get(ticker, {})
                t_obs = t_corr.get("n_observations", 0)
                t_suff = assess_data_sufficiency(t_obs)
                t_sum = get_company_sentiment_summary(ticker)
                t_avg = t_sum.get("avg_sentiment")
                t_news_days = t_df[t_df.get("has_news", 1) == 1]["date"].nunique() if "date" in t_df.columns else 0

                comp_rows.append({
                    "Company": disp,
                    "Headline Count": t_sum.get("total_headlines", 0),
                    "News Days": t_news_days,
                    "Comparable Sessions (N)": t_obs,
                    "Mean Sentiment": get_sentiment_band(t_avg)["label"] if t_avg is not None else "N/A",
                    "Data Status": t_suff["badge"],
                })
            else:
                comp_rows.append({
                    "Company": disp,
                    "Headline Count": 0,
                    "News Days": 0,
                    "Comparable Sessions (N)": 0,
                    "Mean Sentiment": "N/A",
                    "Data Status": "🔴 No Data Available",
                })

        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

# ==============================================================================
# TAB 2: SENTIMENT SHOCK EVENTS
# ==============================================================================
with tab2:
    st.header(f"⚡ Sentiment Shock Event Detection — {selected_display_name}")
    st.markdown(
        "Identifies trading sessions where news sentiment deviated significantly ($\\ge 2.0$ standard deviations) "
        "from the company average, tracking cumulative two-session forward trading returns."
    )

    if all_metrics_df.empty:
        st.info("No daily metrics data loaded.")
    else:
        # Detect shocks across all data
        all_shocks_df = detect_sentiment_shocks(all_metrics_df, min_company_obs=20, threshold_std=2.0)
        filtered_shocks_df = (
            all_shocks_df[all_shocks_df["company"] == target_company]
            if not all_shocks_df.empty and "company" in all_shocks_df.columns
            else pd.DataFrame()
        )

        if filtered_shocks_df.empty:
            st.info(
                f"ℹ️ **No unusual sentiment events were detected for {selected_display_name}.**\n\n"
                "Shock detection requires at least 20 valid news observations and identifies sessions where sentiment differs from the company's average by at least 2 standard deviations."
            )
        else:
            st.plotly_chart(
                plot_sentiment_shocks(all_shocks_df, target_company),
                use_container_width=True,
            )
            st.subheader(f"Detected Shock Events Table — {selected_display_name}")
            st.dataframe(filtered_shocks_df, use_container_width=True)
            st.caption(
                "ℹ️ *Two-session forward return measures cumulative return across the next two trading sessions following the shock event.*"
            )

# ==============================================================================
# TAB 3: AI ANALYST (RAG)
# ==============================================================================
with tab3:
    st.header("🤖 ASK THE FINANCIAL ANALYST")
    st.markdown(
        "Ask questions about financial news sentiment, aggregate statistics, or thematic concerns. "
        "Every response is deterministically routed and strictly grounded in retrieved evidence."
    )

    # Educational RAG Guide
    with st.expander("💡 HOW THE AI ANALYST WORKS"):
        st.markdown(
            """
            ### RAG Engine Architecture & Operational Rules
            
            1. **Your question is interpreted** by a query router.
            2. **Relevant financial evidence is retrieved** from the database.
            3. **Structured statistics are retrieved from SQLite** when numerical answers are required.
            4. **Relevant news headlines are retrieved from ChromaDB** vector search.
            5. **The language model summarizes only the retrieved evidence** without outside guessing.
            6. **Source citations remain visible** at the end of every answer.

            _RAG means Retrieval-Augmented Generation. Retrieval finds the evidence; the language model explains it._
            """
        )

    st.markdown(f"**Suggested Questions for {selected_display_name}:**")
    sample_questions = rag.get_sample_questions_for_company(selected_display_name)
    other_comp = "Infosys" if selected_display_name != "Infosys" else "TCS"
    sample_questions.append({
        "label": f"⚖️ {selected_display_name} vs {other_comp}",
        "query": f"Compare {selected_display_name} and {other_comp} sentiment",
    })
    sq_cols = st.columns(len(sample_questions))
    sample_q = None
    for idx, item in enumerate(sample_questions):
        if sq_cols[idx].button(item["label"]):
            sample_q = item["query"]

    user_query = st.text_input(
        "What would you like to understand?",
        value=sample_q if sample_q else "",
        placeholder=f"e.g. What is the average news sentiment for {selected_display_name}?",
    )

    if st.button("Submit Question", type="primary") or sample_q:
        query_text = sample_q if sample_q else user_query
        if not query_text:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving grounded evidence and compiling response..."):
                result = rag.answer_query(query_text)

            # 4 Pipeline Status Indicators
            emb_avail, _ = rag.is_embedding_model_available()
            ret_success = result.get("retrieval_success", False)
            ret_status = result.get("retrieval_status", "UNKNOWN")
            llm_st = result.get("llm_status", "UNKNOWN")
            llm_success = result.get("llm_success", False)

            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                emb_badge = "🟢 READY" if emb_avail else "🔴 UNAVAILABLE"
                st.markdown(f"**Embedding Model**\n\n`{emb_badge}`")
            with col_s2:
                if ret_success:
                    ret_badge = "🟢 WORKING"
                elif ret_status in ["NO_EVIDENCE_FOUND", "COMPANY_DATA_UNAVAILABLE"]:
                    ret_badge = "🟡 INSUFFICIENT EVIDENCE"
                else:
                    ret_badge = f"🔴 {ret_status}"
                st.markdown(f"**Evidence Retrieval**\n\n`{ret_badge}`")
            with col_s3:
                if llm_success:
                    prov_badge = "🟢 READY"
                elif llm_st == "PROVIDER_NOT_CONFIGURED":
                    prov_badge = "🟡 NOT CONFIGURED"
                elif llm_st == "PROVIDER_ERROR":
                    prov_badge = "🔴 PROVIDER ERROR"
                else:
                    prov_badge = f"⚪ {llm_st}"
                st.markdown(f"**LLM Provider**\n\n`{prov_badge}`")
            with col_s4:
                if llm_success:
                    gen_badge = "🟢 GENUINE LLM EXPLANATION"
                else:
                    gen_badge = "🟡 FACTUAL FALLBACK (GROUNDED)"
                st.markdown(f"**Answer Generation**\n\n`{gen_badge}`")

            st.markdown("---")

            # Provenance & Generation Transparency Banner
            if not llm_success:
                err_detail = result.get("provider_error") or "External LLM is not configured or unavailable."
                st.info(
                    f"ℹ️ **Notice: Factual Database Synthesis Used**\n\n"
                    f"This answer was reconstructed directly from verified SQLite records and ChromaDB headlines "
                    f"because the external language model is unavailable (*{err_detail}*).\n\n"
                    f"• **Numerical Source of Truth:** SQLite pre-aggregated metrics & FinBERT headlines\n"
                    f"• **External LLM Generation:** Inactive"
                )
            else:
                st.success(
                    f"✨ **Genuine AI Explanation Active**: Generated live by **{result.get('provider')}** "
                    f"under the Grounded Answer Contract using verified retrieved evidence."
                )

            # Analyst Response Output
            st.subheader("💡 Analyst Answer")
            st.markdown(result["answer"])

            st.markdown("---")

            # Grounded Evidence & Sources
            if result.get("sources"):
                st.subheader("🔍 Grounded Evidence & Sources")
                st.markdown("**Source Citations:**")
                for src in result["sources"]:
                    st.write(f"- {src}")

            # Technical details expander
            with st.expander("🛠️ Technical Details & Raw Retrieval Inspection"):
                st.write(f"**Target Company:** {result.get('display_name')} ({result.get('company')})")
                st.write(f"**Query Route:** `{result.get('route_type', 'UNKNOWN')}`")
                st.write(f"**LLM Provider Used:** `{result.get('provider')}`")
                st.write(f"**LLM Generation Status:** `{result.get('llm_status')}`")
                st.write(f"**Retrieval Status:** `{result.get('retrieval_status')}`")
                if result.get("provider_error"):
                    st.write(f"**Provider Diagnostic:** `{result.get('provider_error')}`")

                if result.get("structured_evidence"):
                    st.markdown("**Structured Evidence (SQLite Numerical Source of Truth):**")
                    st.json(result["structured_evidence"])
                if result.get("textual_evidence"):
                    st.markdown("**Textual Evidence (Chroma Vector Retrieval):**")
                    st.dataframe(pd.DataFrame(result["textual_evidence"]), use_container_width=True)

# Footer
st.markdown("---")
st.caption(
    "Financial News Sentiment + RAG Analyst | Final Pass Release | "
    "Designed for empirical financial analysis without causal claims."
)
