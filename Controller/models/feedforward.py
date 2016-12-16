import numpy as np
import sys
from models.model import Model
import json


class ModelParams():
    """
    Encapsulates parameters of a model.
    """

    @staticmethod
    def from_dict(data):
        params = ModelParams(
            data["hidden_layers"],
            data["activation"])
        return params

    def __init__(self, hidden_layers, activation):
        self._hidden_layers = hidden_layers
        self._activation = activation

    @property
    def hidden_layers(self):
        return self._hidden_layers

    @property
    def activation(self):
        return self._activation

    def to_dict(self):
        data = {}
        data["hidden_layers"] = self._hidden_layers
        data["activation"] = self._activation
        return data

    def to_string(self):
        return "layers: {}, activation: {}".format(self.hidden_layers, self.activation)


class Network():
    def __init__(self, layer_sizes, model_config):
        self.layer_sizes = layer_sizes
        self.model_config = model_config

        weights = np.array(list(map(float, self.model_config["weights"])))
        l_bound = 0
        r_bound = 0
        self.matrices = []
        for i in range(len(self.layer_sizes) - 1):
            m = self.layer_sizes[i] + 1
            n = self.layer_sizes[i + 1]
            r_bound += m * n
            self.matrices.append(weights[l_bound:r_bound:1].reshape(m, n))
            l_bound = r_bound

    def relu(self, x):
        return np.array([max(0, y) for y in x])

    def tanh(self, x):
        return np.array([np.tanh(y) for y in x])

    def predict(self, input):
        """
        Performs forward pass in the current network instance.
        :param input: Input to the neural network.
        :return: Output of the neural network.
        """
        x = np.array(list(map(float, input["state"])))
        activation_str = self.model_config["activation"]
        activation = ""
        if (activation_str == "relu"):
            activation = self.relu
        if (activation_str == "tanh"):
            activation = self.tanh

        for W in self.matrices:
            x = np.concatenate((x, [1]), axis=0)
            x = activation(np.matmul(x, W))
        result = ""
        assert (self.layer_sizes[-1] == len(x))
        for i in range(len(x)):
            result += str(x[i])
            if (i < self.layer_sizes[-1] - 1):
                result += " "
        return result


class FeedForward(Model):
    """
    Represents a simple feedforward neural network model.
    Can contain multiple networks (for each game phase).
    """

    def __init__(self, game_config, model_config):
        """
        Initializes a new instance of SimpleNN.
        :param game_config: Game configuration file which contains:
        1) input size (for neural network = state of the game
        2) output size (number of "actuators" = number of AI outputs)
        3) number of game phases in total
        :param model_config:
        Model configuration file which contains:
        1) Weights for neural network.
        2) This class name (FeedForward)
        3) this file name (feedforward)
        4) sizes of hidden layers
        """
        super(FeedForward, self).__init__(game_config, model_config)

        phases = self.game_config["game_phases"]
        self.models = []
        for i in range(phases):
            input_size = self.game_config["input_sizes"][i]
            output_size = self.game_config["output_sizes"][i]
            hidden_sizes = list(map(int, self.model_config["hidden_sizes"]))
            layer_sizes = [input_size] + hidden_sizes + [output_size]
            self.models.append(Network(layer_sizes, self.model_config))

    def evaluate(self, input):
        """
        Performs a single forward pass.
        :param input: Input from the game.
        :return: Output of the forward pass.
        """
        curr_phase = int(input["current_phase"])
        return self.models[curr_phase].predict(input)
