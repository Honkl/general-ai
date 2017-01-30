import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from utils.activations import get_activation_tf
from utils.miscellaneous import get_rnn_cell


class QNetwork():
    def __init__(self, hidden_layers, activations):
        self.hidden_layers = hidden_layers
        self.activations = activations
        self.output_size = None
        self.batch_size = None

    def init(self, output_size, batch_size):
        self.output_size = output_size
        self.batch_size = batch_size

    def forward_pass(self, x):
        x = tf.contrib.layers.flatten(x)
        dimensions = self.hidden_layers + [self.output_size]

        for i, (dim, activation) in enumerate(zip(dimensions, [get_activation_tf(x) for x in self.activations])):
            W = tf.get_variable(name="W_{}".format(i),
                                shape=[x.get_shape()[1], dim],
                                initializer=tf.random_normal_initializer(mean=0, stddev=1))
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


class QNetworkRnn():
    def __init__(self, rnn_cell_type, num_units):
        self.output_size = None
        self.rnn_cell_str = rnn_cell_type
        self.num_units = num_units
        self.rnn_cell = get_rnn_cell(rnn_cell_type)(num_units)
        self.state = tf.placeholder(shape=[None], dtype=tf.float32, name="state_rnn")
        self.reuse = False

    def init(self, output_size, batch_size):
        self.output_size = output_size
        self.batch_size = batch_size
        self.state = self.rnn_cell.zero_state(self.batch_size, dtype=tf.float32)

    def forward_pass(self, x):
        with tf.variable_scope("rnn", reuse=True):
            print("Forward pass. x={}".format(x))
            x = tf.contrib.layers.flatten(x)
            print("x.flatten shape: {}".format(x.get_shape()))
            x, self.state = self.rnn_cell(x, self.state)
            print("rnn_output: {}, state: {}".format(x.get_shape(), self.state))
            x = tf_layers.fully_connected(x, self.output_size, activation_fn=get_activation_tf("identity"), scope="FC_1")
        return x

    def to_dictionary(self):
        data = {}
        data["rnn_cell"] = self.rnn_cell_str
        data["num_units"] = self.num_units
        return data
