import math
from collections import namedtuple
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def build_configuration(
    mode="max",
    y_units=r"\%",
    figsize=(640, 480),
    fontsize=20,
    label_pads=(5, 5),
):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["font.family"] = "ptm"
    plt.rcParams["font.serif"] = "phv"
    yellow = (1.0, 0.65, 0.0)
    gray = (0.662, 0.647, 0.576)
    px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
    figsize = (figsize[0] * px, figsize[1] * px)
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
        yellow,
        gray,
        [yellow, "tab:blue"],
        20,
        figsize,
        *label_pads,
        mode,
        y_units,
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


def confusion_matrix(data, filepath=None):
    cmap = "YlGn"
    config = build_configuration(figsize=(480, 480), fontsize=40)
    figure, axis = plt.subplots(figsize=config.figsize)

    counts = data.flatten()
    percentages = data / np.sum(data, 1, keepdims=True)
    percentages = np.round(100 * percentages, 2)

    image = axis.imshow(percentages, cmap, vmin=0, vmax=100)
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    colorbar = axis.figure.colorbar(image, cax=cax, ticks=[0, 25, 50, 75, 100])
    colorbar.outline.set_visible(False)
    colorbar.ax.set_yticklabels([r"0\%", r"25\%", r"50\%", r"75\%", r"100\%"])

    axis.set_xticks(range(data.shape[1]), labels=["0", "1"])
    axis.set_yticks(range(data.shape[0]), labels=["0", "1"])

    # make white grid
    axis.spines[:].set_visible(False)
    axis.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    axis.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    axis.grid(which="minor", color="w", linestyle="-", linewidth=10)
    axis.tick_params(which="minor", bottom=False, left=False)

    percentage_labels = [str(value) + r"\%" for value in percentages.flatten()]
    cell_labels = [f"{v1}\n{v2}" for v1, v2 in zip(counts, percentage_labels)]
    cell_labels = np.asarray(cell_labels).reshape(2, 2)
    textcolors = ["black", "white"]

    kwargs = dict(horizontalalignment="center", verticalalignment="center")
    for col_arg in range(data.shape[0]):
        for row_arg in range(data.shape[1]):
            color = textcolors[int(percentages[col_arg, row_arg] > 50)]
            kwargs.update(color=color)
            text = cell_labels[col_arg, row_arg]
            image.axes.text(row_arg, col_arg, text, **kwargs)

    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    write_or_show(figure, filepath)


def precision_recall_curve(precisions, recalls, best_arg, filepath=None):
    config = build_configuration(
        figsize=(480, 480), fontsize=40, label_pads=(10, 10)
    )

    figure, axis = plt.subplots(figsize=config.figsize)

    axis.set(
        xlabel="Recall",
        xlim=(-0.01, 1.01),
        ylabel="Precision",
        ylim=(-0.01, 1.01),
        aspect="equal",
    )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    set_minor_ticks(axis)
    set_label_pads(axis, config)
    axis.plot(recalls, precisions, "-.", color=config.color_1, linewidth=3)
    # axis.fill_between(
    #     precisions,
    #     recalls,
    #     color=config.color_1,
    #     alpha=0.3,
    # )
    axis.plot(
        precisions[best_arg],
        recalls[best_arg],
        color=config.color_1,
        marker="o",
        markersize=20,
    )
    write_or_show(figure, filepath)


if __name__ == "__main__":
    import paz

    data = paz.logger.load_csv(
        "experiments/10-05-2025_08-25-16_simple_ensemble_0/log.csv"
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
