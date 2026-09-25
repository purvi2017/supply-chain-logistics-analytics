"""Shared chart style so every figure in the project looks consistent."""
import matplotlib.pyplot as plt

BLUE = "#2a78d6"      # primary series
ORANGE = "#eb6834"    # second series / highlight
AQUA = "#1baf7a"
YELLOW = "#eda100"
MUTED = "#b9b8b2"     # de-emphasised bars
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e0"
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def set_style():
    plt.rcParams.update({
        "figure.figsize": (10, 5),
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlecolor": TEXT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": plt.cycler(color=CATEGORICAL),
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "font.size": 10,
        "legend.frameon": False,
        "lines.linewidth": 2,
    })


def label_bars(ax, fmt="{:.1f}%", horizontal=True, pad=0.6):
    """Direct value labels at the end of each bar (text in ink colour, not series colour)."""
    for p in ax.patches:
        if horizontal:
            v = p.get_width()
            ax.text(v + pad, p.get_y() + p.get_height() / 2, fmt.format(v),
                    va="center", ha="left", color=TEXT_2, fontsize=9)
        else:
            v = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2, v + pad, fmt.format(v),
                    va="bottom", ha="center", color=TEXT_2, fontsize=9)
