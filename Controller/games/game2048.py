from games.game import Game
import subprocess
from constants import *
import re


class Game2048(Game):
    def __init__(self, model_condig_file):
        self.model_config_file = model_condig_file

    def run(self):
        command = GAME2048 + PYTHON_SCRIPT + PYTHON_EXE + " \"" + self.model_config_file + "\""
        p = subprocess.Popen(command, stdout=subprocess.PIPE)
        result = p.communicate()[0].decode('ascii')
        result = re.split("\\r\\n|\\n", result)
        return float(result[0].split(":")[1].strip())
