import math
from collections import namedtuple
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt


def build_configuration(mode="max", y_units=r"\%"):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = 20
    plt.rcParams["font.family"] = "ptm"
    plt.rcParams["font.serif"] = "phv"
    # plt.rcParams["legend.loc"] = "upper left"
    yellow = (1.0, 0.65, 0.0)
    gray = (0.662, 0.647, 0.576)
    px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
    figsize = (640 * px, 480 * px)
    Configuration = namedtuple(
        "Configuration",
        [
            "color_1",
            "color_2",
            "palette",
            "fontsize",
            "figsize",
            "x_labelpad",
            "y_labelpad",
            "mode",
            "y_units",
        ],
    )
    return Configuration(
        yellow, gray, [yellow, "tab:blue"], 20, figsize, 5, 5, mode, y_units
    )


def hide_axes(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)


def set_minor_ticks(axis):
    axis.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    axis.xaxis.set_minor_locator(ticker.AutoMinorLocator())


def set_label_pads(axis, config):
    axis.xaxis.labelpad = config.x_labelpad
    axis.yaxis.labelpad = config.y_labelpad


def compute_line_coordinates(data, y_max, config):
    if config.mode == "max":
        x_best = np.argmax(data)
    else:
        x_best = np.argmin(data)
    y_best = data[x_best]
    return x_best, y_best


def set_vertical_line(axis, data, y_max, config):
    x, y = compute_line_coordinates(data, y_max, config)
    y_line = (y / y_max) + 0.05
    y_shift = 0.075 * y_max
    axis.axvline(x, color=config.color_2, linestyle="--", ymax=y_line)
    text = f"{y:.2f}" + f" {config.y_units}"
    x_shift = -2.5
    axis.text(
        x + x_shift,
        y + y_shift,
        text,
        color=config.color_2,
        fontsize=config.fontsize,
    )


def write_or_show(figure, filepath=None):
    if filepath is None:
        plt.show()
    else:
        figure.savefig(filepath, bbox_inches="tight")
        plt.close()


def set_axis(axis, x_label, y_label, x_range=None, y_range=None):
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    if x_range is not None:
        axis.set_xlim(x_range)
    if y_range is not None:
        axis.set_ylim(y_range)


def accuracy(y, filepath=None):
    config = build_configuration()
    figure, axis = plt.subplots(figsize=config.figsize)
    y_max = 100
    data = y_max * np.array(y)
    axis.plot(data, "-o", color=config.color_1)
    set_axis(axis, "Epoch", r"Accuracy (\%)", None, (0, y_max))
    set_vertical_line(axis, data, y_max, config)
    hide_axes(axis)
    set_minor_ticks(axis)
    set_label_pads(axis, config)
    write_or_show(figure, filepath)


def plot_same_axis(axis, ys, y_max, legends, config):
    # ensures first element shows up on top
    for y_arg in reversed(range(len(ys))):
        y = y_max * np.array(ys[y_arg])
        axis.plot(y, "-o", color=config.palette[y_arg])
    # legends must be reversed to match the order of the lines
    if config.mode == "max":
        location = "upper left"
    else:
        location = "upper right"
    axis.legend(legends[::-1], prop={"size": 10}, frameon=False, loc=location)


def accuracies(ys, legends, filepath):
    config = build_configuration("max")
    figure, axis = plt.subplots(figsize=config.figsize)
    y_max = 100
    plot_same_axis(axis, ys, y_max, legends, config)
    set_axis(axis, "Epoch", r"Accuracy (\%)", None, (0, y_max))
    hide_axes(axis)
    set_minor_ticks(axis)
    set_label_pads(axis, config)
    write_or_show(figure, filepath)


def nats_to_bits(nats):
    return np.array(nats) * np.log2(np.e)


def binary_cross_entropy(y, filepath=None):
    config = build_configuration("min", "bits")
    figure, axis = plt.subplots(figsize=config.figsize)
    y_max = min(2.0, math.ceil(max(y)))
    data = nats_to_bits(y)
    axis.plot(data, "-o", color=config.color_1)
    set_axis(axis, "Epoch", "Binary Crossentropy (bits)", None, (0, y_max))
    set_vertical_line(axis, data, y_max, config)
    hide_axes(axis)
    set_minor_ticks(axis)
    set_label_pads(axis, config)
    write_or_show(figure, filepath)


def binary_cross_entropies(ys, legends, filepath=None):
    config = build_configuration("min", "bits")
    figure, axis = plt.subplots(figsize=config.figsize)
    y_max = min(2.0, math.ceil(max(ys[0])))
    ys = [nats_to_bits(y) for y in ys]
    plot_same_axis(axis, ys, y_max, legends, config)
    set_axis(axis, "Epoch", "Binary Crossentropy (bits)", None, (0, y_max))
    # set_vertical_line(axis, data, y_max, config)
    hide_axes(axis)
    set_minor_ticks(axis)
    set_label_pads(axis, config)
    write_or_show(figure, filepath)


if __name__ == "__main__":
    import paz

    data = paz.logger.load_csv(
        "experiments/10-05-2025_08-25-16_ensemble_0/log.csv"
    )
    accuracy(data["val_binary_accuracy"], "accuracy.pdf")
    accuracies(
        [data["val_binary_accuracy"], data["binary_accuracy"]],
        ["Validation", "Train"],
        "accuracies.pdf",
    )

    binary_cross_entropy(data["val_loss"], "loss.pdf")
    binary_cross_entropies(
        [data["val_loss"], data["loss"]],
        ["Validation", "Train"],
        "losses.pdf",
    )
