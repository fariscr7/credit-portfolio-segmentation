"""Stage 5 helpers: sampling and 2-D maps of borrowers (PCA, t-SNE, UMAP).

Colours follow the project palette. Scatter plots use at most three
categorical colours (plus grey) so groups stay distinguishable for
colour-blind readers; ordered bands use one hue from light to dark.
"""

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED
from src.eda import FICO_LABELS, add_value_bands
from src.features import group_purpose
from src.plot_style import SERIES_COLORS, TEXT_MUTED

MAP_SAMPLE_SIZE = 10_000

HOME_COLORS = {
    "MORTGAGE": SERIES_COLORS[0],
    "RENT": SERIES_COLORS[1],
    "OWN": SERIES_COLORS[2],
    "OTHER": TEXT_MUTED,
}
PURPOSE_COLORS = {
    "debt_consolidation": SERIES_COLORS[0],
    "credit_card": SERIES_COLORS[1],
    "other purposes": SERIES_COLORS[2],
}
# One blue from light (low FICO) to dark (high FICO).
FICO_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#184f95", "#0d366b"]


def sample_positions(n_rows, size=MAP_SAMPLE_SIZE, seed=RANDOM_SEED):
    """Sorted random row positions, the same every run (fixed seed)."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=size, replace=False))


def map_labels(sample_loans):
    """The three groupings used to colour the maps, one row per loan."""
    banded = add_value_bands(sample_loans)
    purpose = group_purpose(sample_loans["purpose"])
    purpose = purpose.where(
        purpose.isin(["debt_consolidation", "credit_card"]), "other purposes"
    )
    return pd.DataFrame({
        "home": sample_loans["home_ownership_n"].astype(str).to_numpy(),
        "purpose": purpose.to_numpy(),
        "fico": banded["fico_band"].astype(str).to_numpy(),
    })


def scatter_groups(ax, coords, labels, colors, title):
    """Draw one scatter layer per group, in the order of `colors`."""
    for name, color in colors.items():
        mask = labels == name
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1], s=3, alpha=0.55,
                       color=color, linewidths=0,
                       label=f"{name} ({mask.sum():,})")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.legend(loc="lower left", fontsize=8, markerscale=4,
              handletextpad=0.3, borderaxespad=0.2)


def plot_map_panels(axes, coords, labels, method):
    """Three views of the same 2-D map: by home ownership, purpose, FICO."""
    scatter_groups(axes[0], coords, labels["home"].to_numpy(), HOME_COLORS,
                   f"{method}: home ownership")
    scatter_groups(axes[1], coords, labels["purpose"].to_numpy(),
                   PURPOSE_COLORS, f"{method}: loan purpose")
    scatter_groups(axes[2], coords, labels["fico"].to_numpy(),
                   dict(zip(FICO_LABELS, FICO_RAMP)),
                   f"{method}: FICO band (light = low, dark = high)")
