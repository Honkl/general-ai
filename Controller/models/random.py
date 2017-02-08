from models.model import Model
import numpy as np
import utils.miscellaneous


class Random(Model):
    """
    Represents random model.
    """

    def __init__(self, game):
        """
        Initializes a new instance of Random model for the specified game.
        :param game: Game that will be played.
        """
        self.game_config = utils.miscellaneous.get_game_config(game)

    def evaluate(self, input, current_phase):
        """
        Evaluates model output with the specified input. Output is random.
        :param input: Input to evaluate - there's no any usage of this.
        :param current_phase: Current game phase - there's no any usage of this.
        :return: Random output.
        """
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
        """
        A name of the current model.
        """
        return "random"

    def get_class_name(self):
        """
        A class name of the current model.
        :return:
        """
        return "Random"
