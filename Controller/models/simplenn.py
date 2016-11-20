import numpy as np
from models.model import Model

class Network():
    def __init__(self, layer_sizes, model_config):
        self.layer_sizes = layer_sizes
        self.model_config = model_config

    def predict(self, input):
        """
        Performs forward pass in the current network instance.
        :param input: Input to the neural network.
        :return: Output of the neural network.
        """
        state = np.array(list(map(float, input["state"])))
        weights = np.array(list(map(float, self.model_config["weights"])))
        result = ""
        for i in range(self.layer_sizes[-1]):
            result += str(np.random.random())
            if (i < self.layer_sizes[-1] - 1):
                result += " "
        return result

class SimpleNN(Model):
    def __init__(self, game_config, model_config):
        super(SimpleNN, self).__init__(game_config, model_config)

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