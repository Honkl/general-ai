from games.game import Game
import subprocess
import re

class Alhambra(Game):
    def __init__(self, command):
        self.command = command

    def run(self):
        p = subprocess.Popen(self.command, stdout=subprocess.PIPE)
        result = p.communicate()[0].decode('ascii')
        result = re.split("\\r\\n|\\n", result)

        number_of_players = 3  # TODO: set correct number of players
        status = result[0].split("=")
        if not status:
            raise SyntaxError("Game has ended with error")

        scores = []
        index = 2
        for i in range(number_of_players):
            name = result[index].split('=')[1]
            value = result[index + 1]
            scores.append(float(value))
            index += 2

        print(scores)
        return scores[0]