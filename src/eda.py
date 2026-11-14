"""Stage 3 helpers: value bands, default-rate tables and EDA charts.

`Default` is used here only to DESCRIBE historical outcomes. It is never
an input to the clustering (Stages 4-6).
"""

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.config import TARGET_COLUMN
from src.plot_style import PRIMARY_COLOR, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY

# Value bands: each list of bin edges has one more item than its labels.
# right=False in pd.cut means a band includes its lower edge, e.g. a FICO of
# exactly 700 falls in "700-719" and a loan of exactly $10,000 in "10-15K".
FICO_BINS = [0, 680, 700, 720, 740, 760, np.inf]
FICO_LABELS = ["<680", "680-699", "700-719", "720-739", "740-759", "760+"]
DTI_BINS = [0, 10, 20, 30, 40, np.inf]
DTI_LABELS = ["<10", "10-20", "20-30", "30-40", "40+"]
INCOME_BINS = [0, 40_000, 60_000, 80_000, 100_000, 150_000, np.inf]
INCOME_LABELS = ["<40K", "40-60K", "60-80K", "80-100K", "100-150K", "150K+"]
LOAN_BINS = [0, 5_000, 10_000, 15_000, 20_000, 30_000, np.inf]
LOAN_LABELS = ["<5K", "5-10K", "10-15K", "15-20K", "20-30K", "30K+"]

# Diverging colour map for correlations: blue (-1) -> grey (0) -> red (+1).
CORRELATION_CMAP = LinearSegmentedColormap.from_list(
    "blue_grey_red", ["#256abf", "#f0efec", "#e34948"]
)


def add_value_bands(df):
    """Add banded versions of FICO, DTI, income and loan amount."""
    out = df.copy()
    out["fico_band"] = pd.cut(out["fico_n"], FICO_BINS,
                              labels=FICO_LABELS, right=False)
    out["dti_band"] = pd.cut(out["dti_n"], DTI_BINS,
                             labels=DTI_LABELS, right=False)
    out["income_band"] = pd.cut(out["revenue"], INCOME_BINS,
                                labels=INCOME_LABELS, right=False)
    out["loan_band"] = pd.cut(out["loan_amnt"], LOAN_BINS,
                              labels=LOAN_LABELS, right=False)
    return out


def default_rate_table(df, column, min_loans=0):
    """Loans, defaults and default rate (%) for each value of `column`.

    `lift` = group default rate / portfolio default rate, so 1.5 means
    "50% more likely to default than the average loan".
    Groups with fewer than `min_loans` loans are left out (too small to
    trust).
    """
    overall_rate = df[TARGET_COLUMN].mean()
    table = df.groupby(column, observed=True)[TARGET_COLUMN].agg(
        loans="size", defaults="sum", default_rate="mean",
    )
    table = table[table["loans"] >= min_loans].copy()
    table["lift"] = (table["default_rate"] / overall_rate).round(2)
    table["default_rate"] = (table["default_rate"] * 100).round(1)
    return table


def plot_default_rates(table, title, ax, overall_rate, sort=False):
    """Horizontal bars of default rate (%) with the portfolio average line.

    sort=False keeps the table order (first value at the top), which suits
    ordered bands; sort=True puts the highest rate at the top.
    """
    data = table.sort_values("default_rate") if sort else table.iloc[::-1]
    labels = [str(value) for value in data.index]
    rates = data["default_rate"].to_numpy()

    bar_height = 0.6 if len(rates) > 6 else 0.35   # thin bars, never blocks
    ax.barh(labels, rates, height=bar_height, color=PRIMARY_COLOR)
    ax.axvline(overall_rate, color=TEXT_PRIMARY, linewidth=1, zorder=2)
    for y_position, rate in enumerate(rates):
        # The surface-coloured box keeps the label readable over the line.
        ax.text(rate + 0.4, y_position, f"{rate:.1f}%", va="center",
                fontsize=8, color=TEXT_SECONDARY, zorder=3,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})

    ax.text(overall_rate, 1.0, f" Portfolio {overall_rate:.1f}%",
            transform=ax.get_xaxis_transform(), va="bottom", fontsize=8,
            color=TEXT_SECONDARY)
    ax.set_title(title, pad=16)
    ax.set_xlabel("Default rate (%)")
    ax.set_xlim(0, rates.max() * 1.2)
    ax.grid(axis="y", visible=False)


def plot_correlation_heatmap(corr, ax, title):
    """Annotated heatmap of a correlation matrix on a -1 to +1 scale."""
    image = ax.imshow(corr.to_numpy(), cmap=CORRELATION_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=30,
                  ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    ax.grid(False)
    for row in range(corr.shape[0]):
        for col in range(corr.shape[1]):
            value = corr.iat[row, col]
            text_color = SURFACE if abs(value) > 0.6 else TEXT_PRIMARY
            ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                    fontsize=9, color=text_color)
    ax.set_title(title)
    return image
