# Basic wrapper to start process with any game that has proper interface.

from __future__ import print_function
from __future__ import division

import os
import time

from alhambra import Alhambra
from torcs import Torcs
from mario import Mario
from game2048 import Game2048

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
# prefix = Master directory
prefix = os.path.dirname(os.path.dirname(__location__)) + "\\"  # cut two last directories

PYTHON_EXE = " \"C:\\Anaconda2\\envs\\py3k\\python.exe\""
PYTHON_SCRIPT = " \"" + prefix + "general-ai\\Controller\\script.py\""

MARIO = "java -cp \"" + prefix + "MarioAI\\MarioAI4J\\bin;" + prefix + "MarioAI\\MarioAI4J-Playground\\bin;" + prefix + "MarioAI\\MarioAI4J-Playground\\lib\\*\" mario.GeneralAgent"
GAME2048 = prefix + "2048\\2048\\bin\\Release\\2048.exe"
ALHAMBRA = prefix + "general-ai\\Game-interfaces\\Alhambra\\AlhambraInterface\\AlhambraInterface\\bin\\Release\\AlhambraInterface.exe"

TORCS = "\"" + prefix + "general-ai\\Game-interfaces\\TORCS\\torcs_starter.bat\""
TORCS_XML = " \"" + prefix + "general-ai\\Game-interfaces\\TORCS\\race_config.xml\""
TORCS_JAVA_CP = " \"" + prefix + "general-ai\\Game-interfaces\\TORCS\\scr-client\\classes;" + prefix + "general-ai\\Game-interfaces\\TORCS\\scr-client\\lib\\*\""
#TORCS_EXE_DIRECTORY = " \"C:\\Users\\Jan\\Desktop\\torcs\""  # TODO: Relative path via cmd parameter
TORCS_EXE_DIRECTORY = " \"C:\\Program Files (x86)\\torcs\"" # TODO: Relative path via cmd parameter

torcs_command = TORCS + TORCS_XML + TORCS_JAVA_CP + PYTHON_SCRIPT + TORCS_EXE_DIRECTORY + PYTHON_EXE
alhambra_command = ALHAMBRA + PYTHON_SCRIPT + PYTHON_EXE
game2048_command = GAME2048 + PYTHON_SCRIPT + PYTHON_EXE
mario_command = MARIO + PYTHON_SCRIPT + PYTHON_EXE


if __name__ == '__main__':

    start = time.time()
    game = Torcs(torcs_command)
    #game = Alhambra(alhambra_command)
    #game = Mario(mario_command)
    #game = Game2048(game2048_command)

    print(game.run())
    end = time.time()
    print(end - start)
