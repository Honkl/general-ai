# Basic wrapper to start process with any game that has proper interface.

import numpy as np
from subprocess import call
import sys

def start_game(args):
    call(args)

if (len(sys.argv) != 2):
    print("Not enough arguments (game file path missing)")
else:
    game_exe = sys.argv[1]
    start_game(game_exe)
    print("Finished")
