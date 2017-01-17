from games.game import Game
from constants import *
import json
import subprocess
import socket
import time


class Game2048(Game):
    def __init__(self, model, game_batch_size, seed):
        self.model = model
        self.game_batch_size = game_batch_size
        self.seed = seed

    # === VERSION USING PIPELINENS
    def run(self, advanced_results=False):
        command = "{} {} {}".format(GAME2048, str(self.seed), str(self.game_batch_size))
        p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             bufsize=-1)  # Using PIPEs is not the best solution...

        score = None
        while (True):
            line = p.stdout.readline().decode('ascii')
            if ("SCORE" in line):
                score = line.split(' ')[1]
                break

            result = self.model.evaluate(json.loads(line))
            result = "{}{}".format(result, os.linesep)

            p.stdin.write(bytearray(result.encode('ascii')))
            p.stdin.flush()

        return float(score)

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
