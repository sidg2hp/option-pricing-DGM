"""Publication-quality matplotlib style configuration.

Sets fonts, colours, and figure sizes appropriate for single- and
double-column journal layouts.  Call ``apply_style()`` at the start of
any plotting script.
"""

from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.figure import Figure

STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
    "text.usetex": False,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "figure.figsize": (6.5, 4.5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.5,
    "axes.prop_cycle": cycler(
        "color",
        [
            "#2C3E50", "#E74C3C", "#3498DB", "#2ECC71",
            "#9B59B6", "#F39C12", "#1ABC9C", "#E67E22",
        ],
    ),
}


def apply_style() -> None:
    """Apply the publication style to all subsequent matplotlib figures."""
    plt.rcParams.update(STYLE)


def figure_single_column(height_ratio: float = 0.75) -> Tuple[Figure, Axes]:
    """Return a figure sized for single-column journal layout (3.25 in wide).

    Parameters
    ----------
    height_ratio : float
        Height-to-width ratio.

    Returns
    -------
    fig, ax
    """
    width = 3.25
    fig, ax = plt.subplots(figsize=(width, width * height_ratio))
    return fig, ax


def figure_double_column(height_ratio: float = 0.5) -> Tuple[Figure, Axes]:
    """Return a figure sized for double-column journal layout (6.5 in wide).

    Parameters
    ----------
    height_ratio : float
        Height-to-width ratio.

    Returns
    -------
    fig, ax
    """
    width = 6.5
    fig, ax = plt.subplots(figsize=(width, width * height_ratio))
    return fig, ax


def save_figure(fig: Figure, path_stem: str) -> None:
    """Save a figure as both PDF and PNG.

    Parameters
    ----------
    fig : Figure
    path_stem : str
        File path without extension (e.g. ``"results/fig1"``).
    """
    fig.savefig(f"{path_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{path_stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
