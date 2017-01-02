from games.game import Game
import subprocess
from constants import *
import re


class Alhambra(Game):
    def __init__(self, model_config_file, game_batch_size, seed):
        self.model_config_file = model_config_file
        self.game_batch_size = game_batch_size
        self.seed = seed

    def run(self, advanced_results=False):
        command = ALHAMBRA + PYTHON_SCRIPT + PYTHON_EXE + " \"" + self.model_config_file + "\" " + str(
            self.game_batch_size) + " " + str(self.seed)

        p = subprocess.Popen(command, stdout=subprocess.PIPE)
        result = p.communicate()[0].decode('ascii')
        result = list(filter(None, re.split("\\r\\n|\\n", result)))

        number_of_players = 3  # TODO: set correct number of players
        if advanced_results:
            print(result)
            return list(map(float, result))
        else:
            return float(result[0])
