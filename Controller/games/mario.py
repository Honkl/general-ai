from games.game import Game
import subprocess
import re

class Mario(Game):
    def __init__(self, command):
        self.command = command

    def run(self):
        p = subprocess.Popen(self.command, stdout=subprocess.PIPE)
        result = p.communicate()[0].decode('ascii')
        result = re.split("\\r\\n|\\n", result)

        scores = []
        for line in result:
            if line.startswith("status"):
                for item in line.split(";"):
                    name, value = item.partition("=")[::2]
                    scores.append((name, value))
                    if (name == "passedDistance"):
                        print(value)
                        return float(value)
                break
        return scores