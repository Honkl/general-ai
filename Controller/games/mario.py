from games.game import Game
import subprocess
from constants import *
import json
import platform


class Mario(Game):
    """
    Represents a single Mario game.
    """

    def __init__(self, model, game_batch_size, seed, level=None, vis_on=False, use_visualization_tool=False):
        """
        Initializes a new instance of Mario game.
        :param model: Model which will be playing this game.
        :param game_batch_size: Number of games that will be played immediately (one after one) within the single game
        instance. Result is averaged.
        :param seed: A random seed for random generator within the game.
        :param level: Level for mario game. Can be 'gombas' or 'spikes' for example; this is used in combination with
        use_visualization_tool set to true.
        :param vis_on: Determines whether the Mario will has a visual output. Used in combination with
        use_visualization_tool set to true.
        :param use_visualization_tool: Determines whether use specific visualization tool. Starts different subprocess.
        """
        super(Mario, self).__init__()
        self.model = model
        self.game_batch_size = game_batch_size
        self.seed = seed

        self.use_visualization_tool = use_visualization_tool
        self.vis_on = "0"
        if vis_on:
            self.vis_on = "1"
        self.level = level

    def init_process(self):
        """
        Initializes a subprocess with the game and returns first state of the game.
        """
        windows = platform.system() == "Windows"
        if self.use_visualization_tool:
            params = [MARIO_VISUALISATION, str(self.game_batch_size), str(self.level), str(self.vis_on)]
            if windows:
                command = "{} {} {} {}".format(*params)
            else:
                command = params
        else:
            params = [MARIO, str(self.seed), str(self.game_batch_size)]
            if windows:
                command = "{} {} {}".format(*params)
            else:
                command = params
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        bufsize=-1)  # Using PIPEs is not the best solution...

        data = self.get_process_data()
        return data["state"], data["current_phase"]

    def get_process_data(self):
        """
        Gets a subprocess next data (line).
        :return: a subprocess next data (line).
        """
        line = " "

        # Skip non-json file outputs from mario
        while line == '' or line[0] != '{':
            # print("line: '{}'".format(line))
            line = self.process.stdout.readline().decode('ascii')

        return json.loads(line)
