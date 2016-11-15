from game import Game
import subprocess
import re

class Torcs(Game):
    def __init__(self, command):
        self.command = command

    def run(self):
        p = subprocess.Popen(self.command, stdout=subprocess.PIPE)
        result = p.communicate()[0].decode('ascii')
        result = re.split("\\r\\n|\\n", result)

        for line in result:
            if "RACED DISTANCE:" in line:
                return line.split(":")[1]
        raise SyntaxError("Unknown result from the game (torcs).")