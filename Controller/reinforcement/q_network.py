import tensorflow as tf
from utils.activations import get_activation_tf


class QNetwork():
    def __init__(self, hidden_layers, activations):
        self.hidden_layers = hidden_layers
        self.activations = activations
        self.output_size = None

    def set_output_size(self, size):
        self.output_size = size

    def forward_pass(self, x):
        x = tf.contrib.layers.flatten(x)
        dimensions = self.hidden_layers + [self.output_size]

        for i, (dim, activation) in enumerate(zip(dimensions, [get_activation_tf(x) for x in self.activations])):
            W = tf.get_variable(name="W_{}".format(i),
                                shape=[x.get_shape()[1], dim],
                                initializer=tf.random_normal_initializer())
            h = tf.get_variable(name="h_{}".format(i),
                                shape=[dim],
                                initializer=tf.constant_initializer(0.0))
            x = activation(tf.matmul(x, W) + h)

        return x

    def to_dictionary(self):
        data = {}
        data["hidden_layers"] = self.hidden_layers
        data["activations"] = self.activations
        return data
