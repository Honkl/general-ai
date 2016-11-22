from games.game import Game
import subprocess
import re

class Torcs(Game):
    def __init__(self, command):
        self.command = command

    def run(self):
        p = subprocess.Popen(self.command, stdout=subprocess.PIPE)
        result = p.communicate()[0].decode('ascii')
        result = re.split("\\r\\n|\\n", result)
        distances = []
        for line in result:
            if "RACED DISTANCE:" in line:
                distances.append(line.split(":")[1].strip())
        return float(distances[0])
