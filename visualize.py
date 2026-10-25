"""
Visualization module for Financial News Sentiment + RAG Analyst.
Provides reusable Plotly chart creation functions for Streamlit and analytical reporting.

Statistical Safety:
Chart titles and axis labels strictly avoid causal language (e.g. "Daily Sentiment and Subsequent Trading Return"
rather than "Sentiment Impact on Stock Price"). Rolling correlations are explicitly labeled as observation-based.
"""

from typing import Optional, List
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import logger, get_display_name


def _filter_company_metrics(daily_metrics: pd.DataFrame, company: str) -> pd.DataFrame:
    """Helper to filter daily_metrics by company and sort by date."""
    if daily_metrics.empty or "company" not in daily_metrics.columns:
        return pd.DataFrame()
    df = daily_metrics[daily_metrics["company"] == company].copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df


def plot_price_over_time(daily_metrics: pd.DataFrame, company: str) -> go.Figure:
    """
    Creates a line chart of stock Close price over trading sessions.

    Parameters:
        daily_metrics: DataFrame containing daily metrics or price data.
        company: Company ticker name.
    """
    display_name = get_display_name(company)
    df = _filter_company_metrics(daily_metrics, company)
    fig = go.Figure()

    close_col = "Close" if "Close" in df.columns else ("close" if "close" in df.columns else None)

    if df.empty or close_col is None or df[close_col].isnull().all():
        fig.add_annotation(
            text=f"No price data available for {display_name}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(title=f"STOCK PRICE — {display_name}")
        return fig

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df[close_col],
            mode="lines+markers",
            name="Close Price",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=5),
        )
    )

    fig.update_layout(
        title=f"STOCK PRICE — {display_name}",
        xaxis_title="Trading Session Date",
        yaxis_title="Close Price",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_daily_sentiment_over_time(daily_metrics: pd.DataFrame, company: str) -> go.Figure:
    """
    Creates a bar/line chart of daily mean FinBERT sentiment score.
    Days with no news remain blank (not imputed with zero).
    """
    display_name = get_display_name(company)
    df = _filter_company_metrics(daily_metrics, company)
    fig = go.Figure()

    valid_df = df[df["mean_sentiment"].notna()] if not df.empty and "mean_sentiment" in df.columns else pd.DataFrame()

    if valid_df.empty:
        fig.add_annotation(
            text=f"No sentiment metrics available for {display_name}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(title=f"DAILY NEWS SENTIMENT — {display_name}")
        return fig

    colors = ["#2ca02c" if s >= 0 else "#d62728" for s in valid_df["mean_sentiment"]]

    fig.add_trace(
        go.Bar(
            x=valid_df["date"],
            y=valid_df["mean_sentiment"],
            name="Daily Mean Sentiment",
            marker_color=colors,
            text=valid_df["headline_count"].apply(lambda c: f"Headlines: {c}"),
            hovertemplate="Date: %{x}<br>Sentiment Score: %{y:.4f}<br>%{text}<extra></extra>",
        )
    )

    # Add zero reference line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)

    fig.update_layout(
        title=f"DAILY NEWS SENTIMENT — {display_name}",
        xaxis_title="Trading Session Date",
        yaxis_title="Mean Sentiment Score [-1.0 to +1.0]",
        yaxis=dict(range=[-1.05, 1.05]),
        template="plotly_white",
    )
    return fig


def plot_sentiment_and_returns_over_time(daily_metrics: pd.DataFrame, company: str) -> go.Figure:
    """
    Creates a dual-axis chart comparing daily mean sentiment against subsequent next-session returns.
    """
    display_name = get_display_name(company)
    df = _filter_company_metrics(daily_metrics, company)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    valid_df = (
        df[df["mean_sentiment"].notna() & df["next_session_return"].notna()]
        if not df.empty and "mean_sentiment" in df.columns and "next_session_return" in df.columns
        else pd.DataFrame()
    )

    if valid_df.empty:
        fig.add_annotation(
            text=f"No paired sentiment and return data available for {display_name}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(title=f"Daily Sentiment vs Subsequent Return — {display_name}")
        return fig

    # Trace 1: Sentiment
    fig.add_trace(
        go.Bar(
            x=valid_df["date"],
            y=valid_df["mean_sentiment"],
            name="Mean Sentiment",
            marker_color="#3366cc",
            opacity=0.6,
        ),
        secondary_y=False,
    )

    # Trace 2: Next Session Return
    fig.add_trace(
        go.Scatter(
            x=valid_df["date"],
            y=valid_df["next_session_return"],
            mode="lines+markers",
            name="Next-Session Return",
            line=dict(color="#ff9900", width=2),
            marker=dict(size=6),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=f"Daily Sentiment vs Subsequent Return — {display_name}",
        xaxis_title="Trading Session Date",
        template="plotly_white",
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Mean Sentiment Score [-1.0 to +1.0]", secondary_y=False)
    fig.update_yaxes(title_text="Next-Session Return", secondary_y=True)

    return fig


def plot_sentiment_vs_returns_scatter(daily_metrics: pd.DataFrame, company: str) -> go.Figure:
    """
    Creates a scatter plot comparing mean sentiment score vs next-session return with trendline.
    Uses NumPy for simple linear regression calculation to eliminate statsmodels dependency.
    """
    display_name = get_display_name(company)
    df = _filter_company_metrics(daily_metrics, company)
    if df.empty or "mean_sentiment" not in df.columns or "next_session_return" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for scatter plot", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(title=f"Daily Sentiment vs Subsequent Return Scatter — {display_name}")
        return fig

    valid_df = df[df["mean_sentiment"].notna() & df["next_session_return"].notna()].copy()

    if valid_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No paired valid sentiment and return observations", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(title=f"Daily Sentiment vs Subsequent Return Scatter — {display_name}")
        return fig

    fig = px.scatter(
        valid_df,
        x="mean_sentiment",
        y="next_session_return",
        title=f"Daily Sentiment vs Subsequent Return Scatter — {display_name}",
        labels={
            "mean_sentiment": "Daily Mean Sentiment Score",
            "next_session_return": "Next-Session Trading Return",
        },
        hover_data=["date"] if "date" in valid_df.columns else None,
        template="plotly_white",
    )

    # Calculate linear regression line using NumPy (if >= 2 observations with non-zero variance)
    x = valid_df["mean_sentiment"].values
    y = valid_df["next_session_return"].values

    if len(x) >= 2 and float(np.std(x)) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        x_trend = np.linspace(float(x.min()), float(x.max()), 100)
        y_trend = slope * x_trend + intercept

        fig.add_trace(
            go.Scatter(
                x=x_trend,
                y=y_trend,
                mode="lines",
                name="Trendline (Linear Fit)",
                line=dict(color="red", dash="dash"),
                hoverinfo="skip",
            )
        )

    return fig


def plot_rolling_correlations(daily_metrics: pd.DataFrame, company: str) -> go.Figure:
    """
    Creates a line chart of 7-observation and 14-observation rolling correlations.
    """
    display_name = get_display_name(company)
    df = _filter_company_metrics(daily_metrics, company)
    fig = go.Figure()

    valid_df = (
        df[df["rolling_7_observation_corr"].notna() | df["rolling_14_observation_corr"].notna()]
        if not df.empty and "rolling_7_observation_corr" in df.columns and "rolling_14_observation_corr" in df.columns
        else pd.DataFrame()
    )

    if valid_df.empty:
        fig.add_annotation(
            text=f"Insufficient paired sessions to calculate rolling correlations for {display_name} (minimum 7 required)",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(title=f"Observation-Based Rolling Correlations — {display_name}")
        return fig

    fig.add_trace(
        go.Scatter(
            x=valid_df["date"],
            y=valid_df["rolling_7_observation_corr"],
            mode="lines+markers",
            name="7-observation rolling correlation",
            line=dict(color="#1f77b4", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=valid_df["date"],
            y=valid_df["rolling_14_observation_corr"],
            mode="lines+markers",
            name="14-observation rolling correlation",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title=f"Observation-Based Rolling Correlations — {display_name}",
        xaxis_title="Trading Session Date",
        yaxis_title="Pearson Correlation (r)",
        yaxis=dict(range=[-1.05, 1.05]),
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_sentiment_shocks(shocks_df: pd.DataFrame, company: Optional[str] = None) -> go.Figure:
    """
    Creates a visualization of sentiment shock events (>= 2 SD deviation) and subsequent two-session forward return.
    """
    display_name = get_display_name(company) if company else "All Companies"
    fig = go.Figure()

    if shocks_df.empty:
        fig.add_annotation(
            text="No sentiment shock events detected in dataset",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(title="Sentiment Shock Events")
        return fig

    df = shocks_df.copy()
    if company:
        df = df[df["company"] == company]

    if df.empty:
        fig.add_annotation(
            text=f"No shock events detected for {display_name}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(title=f"Sentiment Shock Events — {display_name}")
        return fig

    colors = ["#d62728" if s < 0 else "#2ca02c" for s in df["mean_sentiment"]]

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["mean_sentiment"],
            mode="markers+text",
            name="Shock Sentiment",
            marker=dict(size=14, color=colors, symbol="diamond"),
            text=(
                df["deviation_in_std"].apply(lambda d: f"{d:.2f} SD")
                if "deviation_in_std" in df.columns
                else df["deviation_from_mean"].apply(lambda d: f"{d:.2f} Diff")
            ),
            textposition="top center",
            hovertemplate=(
                "Company: %{customdata[0]}<br>Date: %{x}<br>Shock Sentiment: %{y:.4f}<br>"
                "Deviation: %{text}<br>Two-Session Forward Return: %{customdata[1]}<extra></extra>"
            ),
            customdata=np.stack((df["company"].apply(get_display_name), df["two_session_forward_return"].apply(lambda r: f"{r:.4f}" if pd.notna(r) else "N/A")), axis=-1),
        )
    )

    fig.update_layout(
        title=f"Detected Sentiment Shock Events (>= 2.0 SD Threshold) — {display_name}",
        xaxis_title="Trading Session Date",
        yaxis_title="Shock Sentiment Score",
        template="plotly_white",
    )
    return fig
