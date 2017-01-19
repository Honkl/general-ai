from models.model import Model
import numpy as np

class Random(Model):

    def __init__(self, game_config):
        self.game_config = game_config


    def evaluate(self, input, current_phase):
        input_sizes = list(map(int, self.game_config["input_sizes"]))
        output_sizes = list(map(int, self.game_config["output_sizes"]))

        assert (input_sizes[current_phase] == len(input))

        result = ""
        for i in range(output_sizes[current_phase]):
            result += str(np.random.random())
            if (i < output_sizes[current_phase] - 1):
                result += " "

        return result

    def get_name(self):
        return "random"

    def get_class_name(self):
        return "Random"