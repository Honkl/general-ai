from games.game import Game
import subprocess
from constants import *
import re


class Mario(Game):
    def __init__(self, model_config_file, game_batch_size, seed):
        self.model_config_file = model_config_file
        self.game_batch_size = game_batch_size
        self.seed = seed

    def run(self, advanced_results=False):
        command = MARIO + PYTHON_SCRIPT + PYTHON_EXE + " \"" + self.model_config_file + "\" " + str(
            self.game_batch_size) + " " + str(self.seed)

        p = subprocess.Popen(command, stdout=subprocess.PIPE)
        result = p.communicate()[0].decode('ascii')
        result = re.split("\\r\\n|\\n", result)

        for line in result:
            if "passed_distance" in line:
                value = float(line.split("=")[1].strip())
                return value