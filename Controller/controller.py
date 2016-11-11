# Basic wrapper to start process with any game that has proper interface.

from __future__ import print_function
from __future__ import division

import os
import subprocess
import re

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
TORCS_EXE_DIRECTORY = " \"C:\\Users\\Jan\\Desktop\\torcs\""  # TODO: Relative path via cmd parameter
# TORCS_EXE_DIRECTORY = " \"C:\\Program Files (x86)\\torcs\"" # TODO: Relative path via cmd parameter

def run_game(command):
    p = subprocess.Popen(command, stdout=subprocess.PIPE)
    result = p.communicate()[0].decode('ascii')
    return re.split("\\r\\n|\\n", result)


def start_torcs():
    command = TORCS + TORCS_XML + TORCS_JAVA_CP + PYTHON_SCRIPT + TORCS_EXE_DIRECTORY + PYTHON_EXE
    result = run_game(command)
    for line in result:
        if "RACED DISTANCE:" in line:
            return line.split(":")[1]
    return []


def start_mario():
    command = MARIO + PYTHON_SCRIPT + PYTHON_EXE
    result = run_game(command)
    scores = []
    for line in result:
        if line.startswith("status"):
            for item in line.split(";"):
                name, value = item.partition("=")[::2]
                scores.append((name, value))
            break
    return scores


def start_2048():
    command = GAME2048 + PYTHON_SCRIPT + PYTHON_EXE
    result = run_game(command)
    return result[0].split(":")[1]


def start_alhambra():
    number_of_players = 3  # TODO:
    command = ALHAMBRA + PYTHON_SCRIPT + PYTHON_EXE
    result = run_game(command)
    status = result[0].split("=")
    if not status:
        print("Game has ended with error")
        return []

    scores = []
    index = 2
    for i in range(number_of_players):
        scores.append((result[index].split('=')[1], result[index + 1]))
        index += 2
    return scores


print(start_mario())
print(start_2048())
print(start_alhambra())
print(start_torcs())
