from collections import namedtuple

import paz

SequentialState = namedtuple("SequentialState", ["add", "call"])


def Sequential(nodes=None):
    if nodes is None:
        nodes = []

    def add(function, *args):
        nodes.append(paz.lock(function, *args))

    def call(x):
        for node in nodes:
            x = node(x)
        return x

    return SequentialState(add, call)
