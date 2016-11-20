# Basic wrapper to start process with any game that has proper interface.

from __future__ import print_function
from __future__ import division

import os
import time
import json
import numpy as np

from games.alhambra import Alhambra
from games.torcs import Torcs
from games.mario import Mario
from games.game2048 import Game2048

np.random.seed(42)
loc = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
# prefix = Master directory
prefix = os.path.dirname(os.path.dirname(loc)) + "\\"  # cut two last directories

PYTHON_EXE = " \"C:\\Anaconda2\\envs\\py3k\\python.exe\""
PYTHON_SCRIPT = " \"" + prefix + "general-ai\\Controller\\script.py\""

MARIO = "java -cp \"" + prefix + "MarioAI\\MarioAI4J\\bin;" + prefix + "MarioAI\\MarioAI4J-Playground\\bin;" + prefix + "MarioAI\\MarioAI4J-Playground\\lib\\*\" mario.GeneralAgent"
GAME2048 = prefix + "2048\\2048\\bin\\Release\\2048.exe"
ALHAMBRA = prefix + "general-ai\\Game-interfaces\\Alhambra\\AlhambraInterface\\AlhambraInterface\\bin\\Release\\AlhambraInterface.exe"

TORCS = "\"" + prefix + "general-ai\\Game-interfaces\\TORCS\\torcs_starter.bat\""
TORCS_XML = " \"" + prefix + "general-ai\\Game-interfaces\\TORCS\\race_config.xml\""
TORCS_JAVA_CP = " \"" + prefix + "general-ai\\Game-interfaces\\TORCS\\scr-client\\classes;" + prefix + "general-ai\\Game-interfaces\\TORCS\\scr-client\\lib\\*\""
PORT = " \"3002\""
#TORCS_EXE_DIRECTORY = " \"C:\\Users\\Jan\\Desktop\\torcs\""  # TODO: Relative path via cmd parameter
TORCS_EXE_DIRECTORY = " \"C:\\Program Files (x86)\\torcs\"" # TODO: Relative path via cmd parameter

# config files for each game (contains I/O sizes)
GAME2048_CONFIG_FILE = prefix + "general-ai\\Game-interfaces\\2048\\2048_config.json"
ALHAMBRA_CONFIG_FILE = prefix + "general-ai\\Game-interfaces\\Alhambra\\Alhambra_config.json"
TORCS_CONFIG_FILE = prefix + "general-ai\\Game-interfaces\\TORCS\\TORCS_config.json"
MARIO_CONFIG_FILE = prefix + "general-ai\\Game-interfaces\\Mario\\Mario_config.json"

# commands used to run games
torcs_command = TORCS + TORCS_XML + TORCS_JAVA_CP + PORT + PYTHON_SCRIPT + TORCS_EXE_DIRECTORY + PYTHON_EXE
alhambra_command = ALHAMBRA + PYTHON_SCRIPT + PYTHON_EXE
game2048_command = GAME2048 + PYTHON_SCRIPT + PYTHON_EXE
mario_command = MARIO + PYTHON_SCRIPT + PYTHON_EXE


def get_init_weights(game_config_file, hidden_sizes):
    with open(game_config_file) as f:
        game_config = json.load(f)
        total_weights = 0
        for phase in range(game_config["game_phases"]):
            input_size = game_config["input_sizes"][phase]
            output_size = game_config["output_sizes"][phase]

            total_weights += input_size * hidden_sizes[0]
            if (len(hidden_sizes) > 1):
                for i in range(len(hidden_sizes) - 1):
                    total_weights += hidden_sizes[i] * hidden_sizes[i + 1]
            total_weights += hidden_sizes[-1] * output_size
        weights = []
        for i in range(total_weights):
            weights.append(np.random.random())
    return weights

if __name__ == '__main__':

    game_config_file = GAME2048_CONFIG_FILE
    #game_config_file = ALHAMBRA_CONFIG_FILE
    #game_config_file = TORCS_CONFIG_FILE
    #game_config_file = MARIO_CONFIG_FILE

    model_config_file = loc + "\\config\\feedforward.json"
    with open(model_config_file, "w") as f:
        hidden_sizes = [32,32]
        data = {}
        data["model_name"] = "feedforward"
        data["class_name"] = "FeedForward"
        data["hidden_sizes"] = hidden_sizes
        data["weights"] = get_init_weights(game_config_file, hidden_sizes)
        f.write(json.dumps(data))


    start = time.time()
    #game = Game2048(game2048_command + " \"" + model_config_file + "\"")
    #game = Alhambra(alhambra_command + " \"" + model_config_file + "\"")
    #game = Torcs(torcs_command + " \"" + model_config_file + "\"")
    #game = Mario(mario_command + " \"" + model_config_file + "\"")


    """ SOME PARALLEL ATTEMPTS """
    xml1 = " \"" + prefix + "general-ai\\Game-interfaces\\TORCS\\race_config_0.xml\""
    xml2 = " \"" + prefix + "general-ai\\Game-interfaces\\TORCS\\race_config_1.xml\""
    port1 = " \"3001\""
    port2 = " \"3002\""

    data = [(xml1, port1), (xml2, port2)]
    import concurrent.futures
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for i in range(100):
            game = Game2048(game2048_command + " \"" + model_config_file + "\"")
            future = executor.submit(game.run)
            results.append(future)

        """
        for (xml, port) in data:
            torcs_command = TORCS + xml + TORCS_JAVA_CP + port + PYTHON_SCRIPT + TORCS_EXE_DIRECTORY + PYTHON_EXE
            print(torcs_command)
            game = Torcs(torcs_command + " \"" + model_config_file + "\"")
            future = executor.submit(game.run)
            results.append(future)
        """

    for i in range(len(results)):
        while not results[i].done():
            time.sleep(100)
        print(results[i].result())
    """"""

    #print(game.run())
    end = time.time()
    print(end - start)
