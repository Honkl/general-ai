from games.game import Game
import subprocess
from constants import *
import json


class Mario(Game):
    def __init__(self, model, game_batch_size, seed):
        self.model = model
        self.game_batch_size = game_batch_size
        self.seed = seed

    def run(self, advanced_results=False):
        command = "{} {} {}".format(MARIO, str(self.seed), str(self.game_batch_size))
        p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             bufsize=-1)  # Using PIPEs is not the best solution...

        score = None
        while (True):
            line = p.stdout.readline().decode('ascii')
            if ("passed_distance" in line):
                score = float(line.split("=")[1].strip())
                break

            if not line[0] == '{':
                # Not a proper JSON file
                continue

            result = self.model.evaluate(json.loads(line))
            result = "{}{}".format(result, os.linesep)

            p.stdin.write(bytearray(result.encode('ascii')))
            p.stdin.flush()

        return score
