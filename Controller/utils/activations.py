import numpy as np


def relu(x):
    return np.array([max(0, y) for y in x])


def tanh(x):
    return np.array([np.tanh(y) for y in x])


def logsig(x):
    return np.array([1 / (1 + np.exp(-y)) for y in x])
