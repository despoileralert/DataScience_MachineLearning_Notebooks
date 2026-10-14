"""Report figures. Each function returns the figure so callers can save or show it."""

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

REGION_COLORS = {"America": "#EF553B", "Asia": "#00CC96", "Europe": "#636EFA"}
HOLIDAY_BAR_COLOR = "#AEC6CF"


def plot_growth_trends(growth: pd.DataFrame) -> plt.Figure:
    """Plot 1: smoothed % change in passengers by region."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        data=growth, x="Date", y="pct_change_smooth", hue="Region",
        linewidth=0.8, palette=REGION_COLORS, ax=ax,
    )
    years = growth["Date"].dt.year
    ax.set_title(f"Smoothed Monthly Air Passengers Trends by Region ({years.min()}-{years.max()})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Percentage Change of Air Passengers (%)")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    return fig


def plot_holiday_relationship(df: pd.DataFrame) -> plt.Figure:
    """Plot 2: passengers vs. holiday count with a linear fit, one panel per region."""
    with sns.axes_style("darkgrid"):
        grid = sns.lmplot(
            data=df, x="HolidayCount", y="Total_OS", col="Region", hue="Region",
            facet_kws={"sharey": False}, palette=REGION_COLORS,
        )
    grid.set_axis_labels("Number of holidays", "Total Overseas Travel")
    grid.figure.suptitle("Relationship between holiday amounts and travel quantities (By region).", y=1.05)
    return grid.figure


def plot_growth_vs_holidays(monthly: pd.DataFrame) -> go.Figure:
    """Plot 3: passenger growth index (line) over average holiday count (bars)."""
    regions = list(monthly["Region"].unique())
    fig = make_subplots(
        rows=1, cols=len(regions), subplot_titles=regions,
        specs=[[{"secondary_y": True}] * len(regions)], shared_yaxes=True,
    )
    for i, region in enumerate(regions, start=1):
        sub = monthly[monthly["Region"] == region]
        fig.add_trace(
            go.Bar(x=sub["Month"], y=sub["HolidayCount"], name=f"{region} Holidays",
                   marker_color=HOLIDAY_BAR_COLOR, opacity=0.5),
            row=1, col=i, secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(x=sub["Month"], y=sub["growth_index"], mode="lines+markers",
                       name=f"{region} Air Travel",
                       line=dict(color=REGION_COLORS.get(region), width=3),
                       marker=dict(size=6, opacity=0.7)),
            row=1, col=i, secondary_y=False,
        )
        fig.update_xaxes(title_text="Month", row=1, col=i)
        fig.update_yaxes(title_text="Air Passenger Index (Base = 100)", row=1, col=i, secondary_y=False)
        fig.update_yaxes(title_text="Number of Holidays", row=1, col=i, secondary_y=True)

    fig.update_layout(
        height=400, width=1600,
        title={"text": "Air Travel Growth and Number of Holidays by Region",
               "x": 0.45, "xanchor": "center", "font": {"size": 25}},
        plot_bgcolor="#f0f0f0", barmode="overlay", showlegend=True,
    )
    return fig
