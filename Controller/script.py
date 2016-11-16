import numpy as np
import os
import json
import sys

np.random.seed(42)


def load_config(path):
    """
    Reads a configuration file using specified path (relative to master directory).
    """
    loc = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    config_file = os.path.dirname(loc) + "\\" + path

    with open(config_file, "r") as file:
        config = json.load(file)
        game_phases = int(config["game_phases"])
        input_sizes = list(map(int, config["input_sizes"]))
        output_sizes = list(map(int, config["output_sizes"]))

    return game_phases, input_sizes, output_sizes


#if (len(sys.argv) != 2):
#    raise Exception("Wrong number of parameters. Insert config file.")

game_info_file = sys.argv[1]
model_info_file = sys.argv[2]

game_phases, input_sizes, output_sizes = load_config(game_info_file)

with open(model_info_file) as model_json:
    config = json.load(model_json)
    name = config["model_name"]
    class_name = config["class_name"]

    imp = __import__(name="models." + name, fromlist=class_name)
    model = getattr(imp, class_name)(config)

while (True):
    line = input()

    if (line == "END"):
        break

    request_data = json.loads(line)
    curr_phase = int(request_data["current_phase"])

    assert(input_sizes[curr_phase] == len(request_data["state"]))

    result = ""
    for i in range(output_sizes[curr_phase]):
        result += str(np.random.random())
        if (i < output_sizes[curr_phase] - 1):
            result += " "

    print(result)
