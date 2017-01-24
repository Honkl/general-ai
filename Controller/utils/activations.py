import numpy as np
import tensorflow as tf


def get_activation(name):
    if name == "relu":
        return relu
    if name == "tanh":
        return tanh
    if name == "logsig":
        return logsig

    raise NotImplementedError


def get_activation_tf(name):
    if name == "relu":
        return tf.nn.relu
    if name == "tanh":
        return tf.nn.tanh
    if name == "identity":
        return tf.identity


def relu(x):
    return np.array([max(0, y) for y in x])


def tanh(x):
    return np.array([np.tanh(y) for y in x])


def logsig(x):
    return np.array([1 / (1 + np.exp(-y)) for y in x])
