"""Stage 8 helpers: compare historical default rates across segments.

`Default` is used here only AFTER the segments were fixed in notebook 04, to
describe how repayment outcomes differed between them. Nothing in this file
changes a segment.
"""

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import chi2_contingency

from src.config import TARGET_COLUMN
from src.eda import default_rate_table
from src.plot_style import TEXT_PRIMARY

Z_95 = 1.96                 # normal quantile for a 95% confidence interval
MIN_CELL_LOANS = 1_000      # smaller groups are too noisy to compare

# One blue from light (low default rate) to dark (high default rate).
RATE_CMAP = LinearSegmentedColormap.from_list(
    "rate_blues", ["#eef4fc", "#5598e7", "#0d366b"]
)


def segment_default_table(loans, names):
    """Default rate, lift and 95% confidence interval for every segment.

    The interval shows how precisely each rate is known: with hundreds of
    thousands of loans per segment it is only about +/- 0.1 points wide.
    """
    table = default_rate_table(loans, "segment")
    rate = table["defaults"] / table["loans"]
    margin = 100 * Z_95 * np.sqrt(rate * (1 - rate) / table["loans"])
    table["ci_low"] = (100 * rate - margin).round(2)
    table["ci_high"] = (100 * rate + margin).round(2)
    table.insert(0, "name", table.index.map(names))
    return table


def association_test(loans, column):
    """Chi-square test of `column` against Default, plus Cramer's V.

    The p-value says whether the differences could be chance. Cramer's V
    (0 = no association, 1 = perfect) says how strong they are.
    """
    counts = pd.crosstab(loans[column], loans[TARGET_COLUMN])
    chi_square, p_value, dof, _ = chi2_contingency(counts)
    n_loans = counts.to_numpy().sum()
    cramers_v = np.sqrt(chi_square / (n_loans * (min(counts.shape) - 1)))
    return {"chi_square": chi_square, "degrees_of_freedom": dof,
            "p_value": p_value, "cramers_v": cramers_v}


def rate_pivot(loans, rows, columns, min_loans=MIN_CELL_LOANS):
    """Default rate (%) for every rows x columns cell.

    Cells with fewer than `min_loans` loans are set to NaN (too noisy).
    """
    grouped = loans.groupby([rows, columns], observed=True)[TARGET_COLUMN]
    rates = grouped.mean().unstack() * 100
    sizes = grouped.size().unstack()
    return rates.where(sizes >= min_loans)


def plot_rate_heatmap(ax, rates, title):
    """Annotated heatmap of default rates (%), light = low, dark = high."""
    values = rates.to_numpy(dtype=float)
    low, high = np.nanmin(values), np.nanmax(values)
    image = ax.imshow(values, cmap=RATE_CMAP, vmin=low, vmax=high,
                      aspect="auto")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            dark_cell = not np.isnan(value) and value > (low + high) / 2
            ax.text(col, row, "–" if np.isnan(value) else f"{value:.1f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if dark_cell else TEXT_PRIMARY)
    ax.set_xticks(range(values.shape[1]), labels=rates.columns)
    ax.set_yticks(range(values.shape[0]), labels=rates.index)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title)
    return image
