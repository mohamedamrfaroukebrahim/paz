from keras.utils import PyDataset
import numpy as np


def wrap(batches, names):
    return dict(zip(names, batches))


class Generator(PyDataset):
    def __init__(self, batcher, data):
        self.data = data
        self.batcher = batcher

    def __len__(self):
        return len(self.data)

    def __getitem__(self, batch_index):
        inputs, labels = self.batcher(self.data[batch_index])
        return inputs, labels
