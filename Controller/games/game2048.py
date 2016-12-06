from games.game import Game
import subprocess
from constants import *
import re


class Game2048(Game):
    def __init__(self, model_config_file, game_batch_size, seed):
        self.model_config_file = model_config_file
        self.game_batch_size = game_batch_size
        self.seed = seed

    def run(self):
        command = GAME2048 + PYTHON_SCRIPT + PYTHON_EXE + " \"" + self.model_config_file + "\" " + str(
            self.game_batch_size) + " " + str(self.seed)

        p = subprocess.Popen(command, stdout=subprocess.PIPE)
        result = p.communicate()[0].decode('ascii')
        result = re.split("\\r\\n|\\n", result)
        return float(result[0].split(":")[1].strip())
