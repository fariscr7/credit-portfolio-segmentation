"""One shared chart style, so every figure in the project looks consistent.

Colours come from a colour-blind-safe reference palette: a single series uses
PRIMARY_COLOR; several categories use SERIES_COLORS in this fixed order.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
]
PRIMARY_COLOR = SERIES_COLORS[0]


def apply_style():
    """Set matplotlib defaults: light surface, hairline grid, quiet axes."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": TEXT_SECONDARY,
        "ytick.labelcolor": TEXT_SECONDARY,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "legend.frameon": False,
        "axes.prop_cycle": plt.cycler(color=SERIES_COLORS),
        "lines.linewidth": 2,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def compact_number(value, _position=None):
    """Format axis ticks as 950, 12K, 3.4M (usable as a tick formatter)."""
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= limit:
            return f"{value / limit:g}{suffix}"
    return f"{value:g}"


def compact_dollars(value, _position=None):
    """Format axis ticks as $950, $12K, $3.4M."""
    return "$" + compact_number(value)


COMPACT_FORMATTER = FuncFormatter(compact_number)
DOLLAR_FORMATTER = FuncFormatter(compact_dollars)
