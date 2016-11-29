from games.game import Game
import subprocess
from constants import *
import re


class Mario(Game):
    def __init__(self, model_config_file):
        self.model_config_file = model_config_file

    def run(self):
        command = MARIO + PYTHON_SCRIPT + PYTHON_EXE + " \"" + self.model_config_file + "\""
        p = subprocess.Popen(command, stdout=subprocess.PIPE)
        result = p.communicate()[0].decode('ascii')
        result = re.split("\\r\\n|\\n", result)

        scores = []
        for line in result:
            if line.startswith("status"):
                for item in line.split(";"):
                    name, value = item.partition("=")[::2]
                    scores.append((name, value))
                    if (name == "passedDistance"):
                        print(value)
                        return float(value)
                break
        return scores
