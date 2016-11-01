import numpy as np
import os
import json

np.random.seed(42)

def load_config(config_file):
    with open (config_file, "r") as file:
        for line in file.readlines():
            line_parts = line.strip().split("=")
            name = line_parts[0]
            value = line_parts[1]
            if name == "game_phases":
                game_phases = int(value)
            elif name == "input_sizes":
                input_sizes = list(map(int, value.split(",")))
            elif name == "output_sizes":
                output_sizes = list(map(int, value.split(",")))
    return game_phases, input_sizes, output_sizes

###################################

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
prefix = os.path.dirname(__location__) + "\\Game-interfaces\\"
game_phases = -1
input_sizes = -1
output_sizes = -1

while (True):
    line = input()

    if (line == "END"):
        break

    if not line.startswith("{"):
        f = prefix + line + "\\" + line + "_config.txt"
        game_phases, input_sizes, output_sizes = load_config(f)
        continue

    # Already in game; received game information, next move is expected from AI
    request_data = json.loads(line)
    curr_phase = int(request_data["current_phase"])

    if (input_sizes[curr_phase] != len(request_data["state"])):
        raise RuntimeError("Wrong number of inputs")

    result = ""
    for i in range(output_sizes[curr_phase]):
        result += str(np.random.random())
        if (i < output_sizes[curr_phase] - 1):
            result += " "

    print(result)
