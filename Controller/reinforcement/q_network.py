import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from utils.activations import get_activation_tf
from utils.miscellaneous import get_rnn_cell


class QNetwork():
    def __init__(self, hidden_layers, activation, dropout_keep=None):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.output_size = None
        self.batch_size = None
        self.dropout = dropout_keep

    def init(self, output_size, batch_size):
        self.output_size = output_size
        self.batch_size = batch_size

    def forward_pass(self, x):
        # Hidden fully connected layers
        for i, dim in enumerate(self.hidden_layers):
            x = tf_layers.fully_connected(inputs=x,
                                          num_outputs=dim,
                                          activation_fn=get_activation_tf(self.activation),
                                          scope="fully_connected_{}".format(i))
            if self.dropout != None:
                x = tf_layers.dropout(x, keep_prob=self.dropout)
        # Output logits
        logits = tf_layers.fully_connected(inputs=x,
                                           num_outputs=self.output_size,
                                           activation_fn=None,
                                           scope="output_layer")

        return logits

    def to_dictionary(self):
        data = {}
        data["hidden_layers"] = self.hidden_layers
        data["activation"] = self.activation
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
            x = tf_layers.fully_connected(x, self.output_size, activation_fn=get_activation_tf("identity"),
                                          scope="FC_1")
        return x

    def to_dictionary(self):
        data = {}
        data["rnn_cell"] = self.rnn_cell_str
        data["num_units"] = self.num_units
        return data
