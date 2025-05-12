from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


wildcard = "experiments/*_ensemble_*/log.csv"

filenames = sorted(glob(wildcard))
accuracies = []
for filename in filenames:
    accuracies.append(pd.read_csv(filename).values[:, 4])

min_num_epochs = np.array([len(accuracy) for accuracy in accuracies]).min()
max_num_epochs = np.array([len(accuracy) for accuracy in accuracies]).max()

accuracies = [accuracy[:min_num_epochs] for accuracy in accuracies]
accuracies = np.array(accuracies)
mean = np.mean(accuracies, axis=0)
stdv = np.std(accuracies, axis=0)

plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 20
plt.rcParams["font.family"] = "ptm"
plt.rcParams["font.serif"] = "phv"

figure, axis = plt.subplots()
yellow = (1.0, 0.65, 0.0)
axis.plot(mean, "-o", color=yellow)

axis.set_ylabel("Accuracy")
axis.set_xlabel("Epochs")
axis.spines["top"].set_visible(False)
axis.spines["right"].set_visible(False)
axis.xaxis.labelpad = 10
axis.yaxis.labelpad = 10
x = list(range(min_num_epochs))
axis.fill_between(x, mean - stdv, mean + stdv, color=yellow, alpha=0.25)

axis.set_ylim([0, 1])
plt.show()
