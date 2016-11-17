from games.game import Game
import subprocess
import re

class Game2048(Game):
    def __init__(self, command):
        self.command = command

    def run(self):
        p = subprocess.Popen(self.command, stdout=subprocess.PIPE)
        result = p.communicate()[0].decode('ascii')
        result = re.split("\\r\\n|\\n", result)
        return result[0].split(":")[1].strip()