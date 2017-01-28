import numpy as np
import json
import utils.activations
import utils.miscellaneous
import lib.simple_esn
from threading import Lock

from models.model import Model


class EchoState(Model):
    state_check_lock = Lock()
    library_esn = None
    echo_state_seed = None

    @staticmethod
    def load_from_file(file_name, game):
        try:
            with open(file_name, "r") as f:
                data = json.load(f)

            weights = data["weights"]
            model = data["model"]
            hidden = list(map(int, model["output_layers"]))
            activation = model["activation"]
            n_readouts = int(model["n_readouts"])
            n_components = int(model["n_components"])
            seed = int(model["echo_state_seed"])
        except:
            raise ValueError("File has wrong format.")

        game_config = utils.miscellaneous.get_game_config(game)
        print("Loading Echo-State model from file {}".format(file_name))
        return EchoState(n_readouts, n_components, hidden, activation, weights, game_config, seed)

    class EchoStateNetwork():
        def __init__(self, layer_sizes, activation, weights):
            self.layer_sizes = layer_sizes
            self.activation = utils.activations.get_activation(activation)
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
            x = np.array(list(map(float, input)))

            # reservoir ESN assume (n_samples, n_features)
            x = EchoState.library_esn.transform(x.reshape(-1, len(input))).flatten()  # we have only one sample
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
        return "echo_state"

    def get_class_name(self):
        return "EchoState"

    def __init__(self, n_readout, n_components, output_layers, activation, weights=None, game_config=None,
                 echo_state_seed=None):
        self.n_readout = n_readout
        self.n_components = n_components
        self.output_layers = output_layers
        self.activation = activation
        self.weights = weights
        self.game_config = game_config

        if EchoState.library_esn == None or echo_state_seed != None:
            if echo_state_seed == None:
                EchoState.echo_state_seed = np.random.randint(0, 2 ** 16)
            else:
                EchoState.echo_state_seed = echo_state_seed

            EchoState.library_esn = lib.simple_esn.SimpleESN(n_readout, n_components,
                                                             random_state=EchoState.echo_state_seed)

        if not weights == None and not game_config == None:
            # Init the network
            phases = self.game_config["game_phases"]
            self.models = []
            used_weights = 0
            output_layers = self.output_layers
            for phase in range(phases):
                input_size = self.game_config["input_sizes"][phase]
                output_size = self.game_config["output_sizes"][phase]
                layer_sizes = [n_readout] + output_layers + [output_size]

                EchoState.state_check_lock.acquire()
                if EchoState.library_esn.weights_ is None:
                    EchoState.library_esn.init_weights(n_samples=1, n_features=input_size)
                EchoState.state_check_lock.release()

                if (phases == 1):
                    self.models.append(self.EchoStateNetwork(layer_sizes, activation, weights))
                else:
                    # slice all weights and use only reliable weights to the current phase
                    new_used_weights = used_weights
                    input_size = n_readout + 1  # bias
                    if len(output_layers) == 0:
                        new_used_weights += input_size * output_size
                    else:
                        new_used_weights += input_size * output_layers[0]
                        for i in range(len(output_layers) - 1):
                            new_used_weights += (output_layers[i] + 1) * output_layers[i + 1]
                        new_used_weights += (output_layers[-1] + 1) * output_size
                    self.models.append(self.EchoStateNetwork(layer_sizes, activation, weights))
                    used_weights = new_used_weights

    def get_new_instance(self, weights, game_config):
        instance = EchoState(self.n_readout, self.n_components, self.output_layers, self.activation, weights,
                             game_config)
        return instance

    def get_number_of_parameters(self, game):
        """
        Evaluates number of parameters of neural networks (e.q. weights of network).
        :return: Number of parameters of neural network (this will be equal to evolution individual length).
        """
        game_config = utils.miscellaneous.get_game_config(game)
        total_weights = 0
        learnable_layers = self.output_layers
        for phase in range(game_config["game_phases"]):
            input_size = self.n_readout + 1  # bias
            output_size = game_config["output_sizes"][phase]
            if len(learnable_layers) == 0:
                total_weights += input_size * output_size
            else:
                total_weights += input_size * learnable_layers[0]
                for i in range(len(learnable_layers) - 1):
                    total_weights += (learnable_layers[i] + 1) * learnable_layers[i + 1]
                total_weights += (learnable_layers[-1] + 1) * output_size
        return total_weights

    def evaluate(self, input, current_phase):
        """
        Performs a single forward pass.
        :param input: Input from the game.
        :return: Output of the forward pass.
        """
        return self.models[current_phase].predict(input)

    def to_string(self):
        """
        A string representation of the current object, that describes parameters.
        :return: A string representation of the current object.
        """
        return "ESN - echo-state-size: {}, n_readouts: {}, output_layers: {}, activation: {}".format(self.n_components,
                                                                                                     self.n_readout,
                                                                                                     self.output_layers,
                                                                                                     self.activation)

    def to_dictionary(self):
        """
        Creates dictionary representation of model parameters.
        :return: Dictionary of model parameters.
        """
        data = {}
        data["n_readouts"] = self.n_readout
        data["n_components"] = self.n_components
        data["output_layers"] = self.output_layers
        data["activation"] = self.activation
        data["echo_state_seed"] = EchoState.echo_state_seed
        return data
