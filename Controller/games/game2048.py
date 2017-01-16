from games.game import Game
import subprocess
from constants import *
import json


class Game2048(Game):
    def __init__(self, model, game_batch_size, seed):
        self.model = model
        self.game_batch_size = game_batch_size
        self.seed = seed

    def run(self, advanced_results=False):
        command = "{} {} {}".format(GAME2048, str(self.seed), str(self.game_batch_size))
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, bufsize=1)

        score = None
        while (True):
            std = p.stdout.readline().decode('ascii')

            if ("SCORE" in std):
                score = std.split(' ')[1]
                break

            result = self.model.evaluate(json.loads(std))
            result = "{}{}".format(result, os.linesep)
            p.stdin.write(result.encode('ascii'))
            p.stdin.flush()

        return float(score)
