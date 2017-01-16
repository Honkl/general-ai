import numpy as np
import constants
import json


class MLP():
    """
    Represents a simple feedforward MLP neural network model.
    Can contain multiple networks, each one for each game phase. Contains instances of 'MLPNetwork'.
    """

    class MLPNetwork():
        def __init__(self, layer_sizes, activation, weights):
            self.layer_sizes = layer_sizes
            self.activation = activation
            self.weights = weights

            weights = np.array(list(map(float, self.weights)))
            l_bound = 0
            r_bound = 0
            self.matrices = []
            for i in range(len(self.layer_sizes) - 1):
                m = self.layer_sizes[i] + 1
                n = self.layer_sizes[i + 1]
                r_bound += m * n
                self.matrices.append(weights[l_bound:r_bound:1].reshape(m, n))
                l_bound = r_bound

        def predict(self, input):
            """
            Performs forward pass in the current network instance.
            :param input: Input to the neural network.
            :return: Output of the neural network.
            """
            x = np.array(list(map(float, input["state"])))
            for W in self.matrices:
                x = np.concatenate((x, [1]), axis=0)
                x = self.activation(np.matmul(x, W))

            result = ""
            assert (self.layer_sizes[-1] == len(x))
            x = self.normalize(x)
            for i in range(len(x)):
                result += str(x[i])
                if (i < self.layer_sizes[-1] - 1):
                    result += " "
            return result

        def normalize(self, x):
            """
            Normalizes the specified interval to [0, 1].
            :param x: Values to be normalized.
            :return: Normalized [0, 1] interval.
            """
            min_val = min(x)
            max_val = max(x)
            if max_val - min_val == 0:
                return x
            return np.array([((x_i - min_val) / (max_val - min_val)) for x_i in x])

    def get_name(self):
        return "mlp"

    def get_class_name(self):
        return "MLP"

    def __init__(self, hidden_layers, activation, weights=None, game_config=None):
        """
        Initializes a new instance of SimpleNN.
        :param hidden_layers: list of sizes of hidden layers.
        :param activation: activation function.
        :param weights: Weights of the current network instance.
        :param game_config: Game configuration dictionary, consists of:
        1) input size (for neural network = state of the game
        2) output size (number of "actuators" = number of AI outputs)
        3) number of game phases in total
        (This parameter is dictionary [json]).
        """
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.weights = weights
        self.game_config = game_config

    def get_new_instance(self, weights, game_config):
        instance = MLP(self.hidden_layers, self.activation, weights, game_config)
        return instance

    def get_number_of_weights(self, game):
        """
        Evaluates number of parameters of neural networks (e.q. weights of network).
        :return: Number of parameters of neural network (this will be equal to evolution individual length).
        """
        game_config_file = ""
        if game == "alhambra":
            game_config_file = constants.ALHAMBRA_CONFIG_FILE
        if game == "2048":
            game_config_file = constants.GAME2048_CONFIG_FILE
        if game == "mario":
            game_config_file = constants.MARIO_CONFIG_FILE
        if game == "torcs":
            game_config_file = constants.TORCS_CONFIG_FILE

        with open(game_config_file) as f:
            game_config = json.load(f)
            total_weights = 0
            h_sizes = self.hidden_layers
            for phase in range(game_config["game_phases"]):
                input_size = game_config["input_sizes"][phase] + 1
                output_size = game_config["output_sizes"][phase]
                total_weights += input_size * h_sizes[0]
                if (len(h_sizes) > 1):
                    for i in range(len(h_sizes) - 1):
                        total_weights += (h_sizes[i] + 1) * h_sizes[i + 1]
                total_weights += (h_sizes[-1] + 1) * output_size
        return total_weights

    def evaluate(self, input):
        """
        Performs a single forward pass.
        :param input: Input from the game.
        :param weights: Weights for the network (individual from evolution).
        :return: Output of the forward pass.
        """
        curr_phase = int(input["current_phase"])
        phases = self.game_config["game_phases"]
        self.models = []
        used_weights = 0
        hidden_layers = self.hidden_layers
        for phase in range(phases):
            input_size = self.game_config["input_sizes"][phase]
            output_size = self.game_config["output_sizes"][phase]
            layer_sizes = [input_size] + hidden_layers + [output_size]
            if (phases == 1):
                self.models.append(self.MLPNetwork(layer_sizes, self.activation, self.weights))
            else:
                # slice all weights and use only reliable weights to the current phase
                new_used_weights = used_weights
                input_size = input_size + 1  # bias
                new_used_weights += input_size * hidden_layers[0]
                for i in range(len(hidden_layers) - 1):
                    new_used_weights += (hidden_layers[i] + 1) * hidden_layers[i + 1]
                new_used_weights += (hidden_layers[-1] + 1) * output_size
                self.models.append(self.MLPNetwork(layer_sizes=layer_sizes,
                                                   activation=self.activation,
                                                   weights=self.weights[used_weights:new_used_weights]))
                used_weights = new_used_weights

        return self.models[curr_phase].predict(input)

    def to_string(self):
        """
        A string representation of the current object, that describes parameters.
        :return: A string representation of the current object.
        """
        return "layers: {}, activation: {}".format(self.hidden_layers, self.activation)

    def to_dictionary(self):
        """
        Creates dictionary representation of model parameters.
        :return: Dictionary of model parameters.
        """
        data = {}
        data["hidden_layers"] = self.hidden_layers
        data["activation"] = self.activation
        return data
