import numpy as np
import os
import json
import sys

np.random.seed(42)


def load_config(path):
    loc = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    config_file = os.path.dirname(loc) + "\\" + path

    game_phases = -1
    input_sizes = -1
    output_sizes = -1
    with open(config_file, "r") as file:

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

    if (game_phases == -1 or input_sizes == -1 or output_sizes == -1):
        print("Invalid configuration file.")
        exit()

    return game_phases, input_sizes, output_sizes


if (len(sys.argv) != 2):
    print("Wrong number of parameters. Insert config file.")
    exit()

game_phases, input_sizes, output_sizes = load_config(sys.argv[1])

while (True):
    line = input()

    if (line == "END"):
        break

    # Already in game; received game information, next move is expected from AI
    request_data = json.loads(line)
    curr_phase = int(request_data["current_phase"])

    if (input_sizes[curr_phase] != len(request_data["state"])):
        raise RuntimeError("Wrong number of inputs", input_sizes[curr_phase], len(request_data["state"]))

    result = ""
    for i in range(output_sizes[curr_phase]):
        result += str(np.random.random())
        if (i < output_sizes[curr_phase] - 1):
            result += " "

    print(result)
