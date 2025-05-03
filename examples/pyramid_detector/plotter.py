import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib
import matplotlib.ticker


def histogram(
    data: np.ndarray,
    title: str = "Distribution",
    xlabel: str = "Value",
    ylabel: str = "Frequency",
) -> plt.Figure:
    try:
        plt.rcParams["text.usetex"] = True
    except matplotlib.pyplot.rcParams.NotImplementedError:
        plt.rcParams["text.usetex"] = False

    fig, ax = plt.subplots(figsize=(8, 5))

    # Determine unique values and set bins to display each unique value
    unique_values = np.unique(data)
    if len(unique_values) > 0:
        # Create bins centered around each unique integer value
        bins = np.append(unique_values - 0.5, unique_values[-1] + 0.5)
    else:
        bins = 10  # Default bins if no data

    # Use a single color for the bars and set the edge color, adjust rwidth for bar width
    n, bins, patches = ax.hist(
        data,
        bins=bins,
        color=(1.0, 0.65, 0.0),
        edgecolor=(0.662, 0.647, 0.576),
        alpha=0.9,
        zorder=2,
        rwidth=0.98,
    )  # Increased rwidth

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()

    # Turn on minor ticks for both axes initially
    ax.minorticks_on()

    # Set major ticks to be the unique values and label them
    if len(unique_values) > 0:
        ax.set_xticks(unique_values)
        ax.set_xticklabels(unique_values)
    else:
        ax.set_xticks([])  # No ticks if no data

    # Configure tick parameters for major ticks
    ax.tick_params(which="major", length=5, color="gray", width=0.8)
    ax.tick_params(axis="both", which="both", pad=5)  # Add some padding

    # Configure tick parameters for minor ticks, turn off for x-axis
    ax.tick_params(axis="y", which="minor", length=2, color="gray", width=0.5)
    ax.tick_params(
        axis="x", which="minor", bottom=False, top=False
    )  # Turn off minor ticks on x-axis

    # Ensure minor ticks on x-axis are completely off
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    # Ensure major ticks on y-axis are handled by default locator if not explicitly set
    ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    ax.grid(axis="y", linestyle="--", alpha=0.6, color="lightgray", zorder=1)

    ax.set_title(r"\textbf{" + title + r"}", fontsize=16)
    ax.set_xlabel(r"\textbf{" + xlabel + r"}", fontsize=12)
    ax.set_ylabel(r"$\mathbf{" + ylabel + r"}$", fontsize=12)

    fig.tight_layout()

    return fig
