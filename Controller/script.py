import numpy as np
import os
import json
import sys

np.random.seed(42)

if (len(sys.argv) != 3):
    raise Exception("Wrong number of parameters. Insert config file.")

loc = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
game_info_file = os.path.dirname(loc) + "\\" + sys.argv[1]

model_info_file = sys.argv[2]

with open(game_info_file, "r") as f:
    game_config = json.load(f)

with open(model_info_file) as f:
    model_config = json.load(f)
    name = model_config["model_name"]
    class_name = model_config["class_name"]

    imp = __import__(name="models." + name, fromlist=class_name)
    model = getattr(imp, class_name)(game_config, model_config)

while (True):
    line = input()

    if (line == "END"):
        break

    result = model.evaluate(json.loads(line))
    print(result)
