from games.game import Game
import subprocess
from constants import *
import json


class Alhambra(Game):
    """
    Represents single Alhambra game. Provides communication with internal game process.
    """

    def __init__(self, model, game_batch_size, seed):
        """
        Initializes a new instance of Alhambra game.
        :param model: Model which will be playing this game.
        :param game_batch_size: Number of games that will be played immediately (one after one) within the single game
        instance. Result is averaged.
        :param seed: A random seed for random generator within the game.
        """
        super(Alhambra, self).__init__()
        self.model = model
        self.game_batch_size = game_batch_size
        self.seed = seed

    def init_process(self):
        """
        Initializes a subprocess with the game and returns first state of the game.
        """
        command = "{} {} {}".format(ALHAMBRA, str(self.seed), str(self.game_batch_size))
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        bufsize=-1)  # Using PIPEs is not the best solution...

        data = self.get_process_data()
        return data["state"], data["current_phase"]

    def get_process_data(self):
        """
        Gets a subprocess next data (line).
        :return: a subprocess next data (line).
        """
        line = self.process.stdout.readline().decode('ascii')
        return json.loads(line)
