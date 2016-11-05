# Basic wrapper to start process with any game that has proper interface.

from __future__ import print_function
from __future__ import division

import os
from subprocess import call

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
# prefix = Master directory
prefix = os.path.dirname(os.path.dirname(__location__)) + "\\" # cut two last directories

MARIO = "java -cp \"" + prefix + "MarioAI\\MarioAI4J\\bin;" + prefix + "MarioAI\\MarioAI4J-Playground\\bin;" + prefix + "MarioAI\\MarioAI4J-Playground\\lib\\*\" mario.GeneralAgent"
GAME2048 = prefix + "2048\\2048\\bin\\Debug\\2048.exe"
ALHAMBRA = prefix + "general-ai\\Game-interfaces\\Alhambra\\AlhambraInterface\\AlhambraInterface\\bin\\Debug\\AlhambraInterface.exe"

TORCS = "\"" + prefix + "general-ai\\Game-interfaces\\TORCS\\torcs_starter.bat\""
TORCS_XML = " \"" + prefix + "general-ai\\Game-interfaces\\TORCS\\race_config.xml\""
TORCS_JAVA_CP = " \"" + prefix + "general-ai\\Game-interfaces\\TORCS\\scr-client\\classes;" + prefix + "general-ai\\Game-interfaces\\TORCS\\scr-client\\lib\\*\""
TORCS_PYTHON_SCRIPT = " \"" + prefix + "general-ai\\Controller\\script.py\""
TORCS_EXE_DIRECTORY = " \"C:\\Users\\Jan\\Desktop\\torcs\"" # TODO: Relative path via cmd parameter

def start_torcs():
    command = TORCS + TORCS_XML + TORCS_JAVA_CP + TORCS_PYTHON_SCRIPT + TORCS_EXE_DIRECTORY
    call(command)

def start_mario():
    call(MARIO)

def start_2048():
    call(GAME2048)

def start_alhambra():
    call(ALHAMBRA)


#start_mario()
#start_2048()
#start_alhambra()
start_torcs()

print("Finished")
