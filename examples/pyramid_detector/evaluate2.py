import numpy as np
import sklearn

import plot

y_pred_ensembles = np.load(
    "results/14-05-2025_17-31-45_ensemble/y_model_pred.npy"
)
y_true = np.load("results/14-05-2025_17-31-45_ensemble/y_data.npy")
use_ensemble = True

if use_ensemble:
    y_pred_scores = np.mean(y_pred_ensembles, axis=0)
else:
    y_pred_scores = y_pred_ensembles[0]

precisions, recalls, thresholds = sklearn.metrics.precision_recall_curve(
    y_true, y_pred_scores
)

f1_score = 2 * (precisions * recalls) / (precisions + recalls)
best_arg = np.argmax(f1_score)
plot.precision_recall_curve(precisions, recalls, best_arg)
threshold = thresholds[best_arg]
# print(sklearn.metrics.confusion_matrix(y_true, np.mean(y_pred, axis=0) > 0.5))
print("Best selected threshold: ", threshold)
y_pred = y_pred_scores > threshold

data = sklearn.metrics.confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = data.flatten()


accuracy = (tp + tn) / (tp + fp + tn + fn)
recall = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)
false_positive_rate = fp / (fp + tn)

print(f"Accuracy: {100 * accuracy:.2f}%")
print(
    f"Recall (True Positive Rate): Of all actual positive instances, classifier correctly identifies {100 * recall:.2f}%."
)

print(
    f"False Alarm Rate (False Positive Rate): {100 * false_positive_rate:.2f}%"
)

print(
    f"Specificity: Of all actual negative instances, classifier correctly identifies {100 * specificity:.2f}%."
)

print(
    f"Precision: When classifier predicts an instance is positive, it is correct {100 * precision:.2f}% of the time."
)
plot.confusion_matrix(data)
config = plot.build_configuration()

accuracies = []
for num_ensembles in range(len(y_pred_ensembles)):
    y_pred_scores = np.mean(y_pred_ensembles[0 : (num_ensembles + 1)], axis=0)
    precisions, recalls, thresholds = sklearn.metrics.precision_recall_curve(
        y_true, y_pred_scores
    )
    f1_score = 2 * (precisions * recalls) / (precisions + recalls)
    threshold = thresholds[np.argmax(f1_score)]
    y_pred = y_pred_scores > threshold
    data = sklearn.metrics.confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = data.flatten()
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    accuracies.append(accuracy)
accuracies = np.array(accuracies)

import plot
import matplotlib.pyplot as plt

config = plot.build_configuration(figsize=(640, 480))
figure, axis = plt.subplots(figsize=config.figsize)
axis.plot(accuracies, "-o", linewidth=2, color=config.color_1)
plot.hide_axes(axis)
plot.set_minor_ticks(axis)
plot.write_or_show(figure)
