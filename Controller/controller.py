# Basic wrapper to start process with any game that has proper interface.

from __future__ import print_function
from __future__ import division

import os
from subprocess import call

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
prefix = os.path.join(__location__, "../../")

MARIO = "java -cp \"" + prefix + "MarioAI/MarioAI4J/bin;" + prefix + "MarioAI/MarioAI4J-Playground/bin;" + prefix + "MarioAI/MarioAI4J-Playground/lib/*\" mario.GeneralAgent"
GAME2048 = prefix + "2048/2048/bin/Debug/2048.exe"

call(MARIO)
call(GAME2048)
print("Finished")
