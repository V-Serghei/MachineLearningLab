"""Shared utilities for MachineLearningLab scripts."""
import os

import matplotlib.pyplot as plt

_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def save_figure(name: str, dpi: int = 150) -> None:
    """Save the current matplotlib figure to results/<name>.png.

    Args:
        name: Base filename (without extension).
        dpi:  Output resolution in dots per inch.
    """
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(_RESULTS_DIR, f"{name}.png"), dpi=dpi, bbox_inches="tight")
