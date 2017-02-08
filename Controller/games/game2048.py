from games.game import Game
from constants import *
import json
import subprocess
import platform
import socket
import time


class Game2048(Game):
    """
    Represents a single 2048 game.
    """

    def __init__(self, model, game_batch_size, seed, use_advanced_tool=False):
        """
        Initializes a new instance of 2048 game.
        :param model: Model which will be playing this game.
        :param game_batch_size: Number of games that will be played immediately (one after one) within the single game
        instance. Result is averaged.
        :param seed: A random seed for random generator within the game.
        :param use_advanced_tool: True if some advanced results are wanted (using different process).
        """
        super(Game2048, self).__init__()
        self.model = model
        self.game_batch_size = game_batch_size
        self.seed = seed

        self.use_advanced_tool = use_advanced_tool

    def init_process(self):
        """
        Initializes a subprocess with the game and returns first state of the game.
        """
        windows = platform.system() == "Windows"
        if self.use_advanced_tool:
            params = [GAME2048_ADVANCED_TOOL, str(self.seed), str(self.game_batch_size)]
            if windows:
                command = "{} {} {}".format(*params)
            else:
                command = ["mono"] + params
        else:
            params = [GAME2048, str(self.seed), str(self.game_batch_size)]
            if windows:
                command = "{} {} {}".format(*params)
            else:
                command = ["mono"] + params

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

    """
    #==== VERSION USING SOCKETS INSTEAD OF PIPELINES
    def run(self, advanced_results=False):
        sock = socket.socket()
        sock.bind(("localhost", 0))  # let OS determine free port
        port = sock.getsockname()[1]
        sock.listen(1)

        command = "{} {} {} {}".format(GAME2048, str(self.seed), str(self.game_batch_size), str(port))
        subprocess.Popen(command)

        connection, _ = sock.accept()

        score = None

        while (True):
            line = self.receive(connection)

            if ("SCORE" in line):
                score = line.split(' ')[1]
                break

            result = self.model.evaluate(json.loads(line))
            result = "{}{}".format(result, os.linesep)

            connection.sendall(bytearray(result.encode('ascii')))

        connection.close()
        sock.close()
        return float(score)

    def receive(self, connection):
        BUFF_SIZE = 4096  # 4 KiB
        data = ""
        while True:
            part = connection.recv(BUFF_SIZE)
            data += part.decode('ascii')
            if len(part) < BUFF_SIZE:
                # either 0 or end of data
                break
        return data
    """
