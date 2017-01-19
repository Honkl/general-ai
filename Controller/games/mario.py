from games.game import Game
import subprocess
from constants import *
import json


class Mario(Game):
    def __init__(self, model, game_batch_size, seed):
        super(Mario, self).__init__()
        self.model = model
        self.game_batch_size = game_batch_size
        self.seed = seed

    def init_process(self):
        """
        Initializes a subprocess with the game and returns first state of the game.
        """
        command = "{} {} {}".format(MARIO, str(self.seed), str(self.game_batch_size))
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        bufsize=-1)  # Using PIPEs is not the best solution...

        data = self.get_process_data()
        return data["state"], data["current_phase"]

    def get_process_data(self):
        line = " "

        # Skip non-json file outputs from mario
        while line == '' or line[0] != '{':
            # print("line: '{}'".format(line))
            line = self.process.stdout.readline().decode('ascii')

        return json.loads(line)
