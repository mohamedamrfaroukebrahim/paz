import jax.numpy as jp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.ticker


def histogram(data, xlabel, ylabel="Frequency"):
    gray = (0.662, 0.647, 0.576)
    yellow = ((1.0, 0.65, 0.0),)
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = 20
    plt.rcParams["font.family"] = "ptm"
    plt.rcParams["font.serif"] = "phv"

    figure, axis = plt.subplots()

    axis.hist(data, color=yellow)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)

    mean = jp.mean(data)
    axis.axvline(mean, color=gray, linestyle="--", label="Mean", ymax=0.7)
    y_text = 0.725 * axis.get_ylim()[1]
    axis.text(mean - 1.0, y_text, "Mean: {:.2f}".format(mean), color=gray)

    axis.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axis.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
    axis.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    axis.set_xlabel(xlabel, labelpad=20)
    axis.set_ylabel(ylabel, labelpad=20)
    figure.tight_layout()
    return figure


def histogram_uniques(data, xlabel, ylabel="Frequency"):
    gray = (0.662, 0.647, 0.576)
    yellow = ((1.0, 0.65, 0.0),)
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = 20
    plt.rcParams["font.family"] = "ptm"
    plt.rcParams["font.serif"] = "phv"

    figure, axis = plt.subplots()

    unique_values = np.unique(data)
    bins = np.append(unique_values - 0.5, unique_values[-1] + 0.5)

    axis.hist(data, bins=bins, color=yellow)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)

    mean = jp.mean(data)
    axis.axvline(mean, color=gray, linestyle="--", label="Mean", ymax=0.7)
    y_text = 0.725 * axis.get_ylim()[1]
    axis.text(mean - 1.0, y_text, "Mean: {:.2f}".format(mean), color=gray)

    axis.set_xticks(unique_values)
    axis.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axis.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
    axis.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    axis.set_xlabel(xlabel, labelpad=20)
    axis.set_ylabel(ylabel, labelpad=20)
    figure.tight_layout()
    return figure
