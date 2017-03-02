"""
This file contains important constants. Directories are relative to current master directory. Torcs installation folder
must be provided in 'install_directory.txt' file.
"""

import os
import platform

loc = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
prefix = os.path.dirname(os.path.dirname(loc)) + "/"  # cut two last directories; prefix = master directory

delimiter = ";" if (platform.system() == "Windows") else ":"

# MARIO STUFF
MARIO_CP = prefix + "MarioAI/MarioAI4J/bin" + delimiter + prefix + "MarioAI/MarioAI4J-Playground/bin" + delimiter + prefix + "MarioAI/MarioAI4J-Playground/lib/*"
MARIO_CLASS = "mario.GeneralAgent"
MARIO_VISUALISATION_CLASS = "mario.VisualizationTool"

# GAME 2048 STUFF
GAME2048_PY_PATH = prefix + "general-ai/Game-interfaces/Game2048/game_2048.py"

# ALHAMBRA STUFF
ALHAMBRA = prefix + "general-ai/Game-interfaces/Alhambra/AlhambraInterface/AlhambraInterface/bin/Release/AlhambraInterface.exe"

# TORCS STUFF
TORCS_BAT = "\"" + prefix + "general-ai/Game-interfaces/TORCS/torcs_starter.bat\""
TORCS_VIS_ON_BAT = "\"" + prefix + "general-ai/Game-interfaces/TORCS/torcs_starter_vis_on.bat\""
TORCS_XML = " \"" + prefix + "general-ai/Game-interfaces/TORCS/race_config.xml\""
TORCS_JAVA_CP = " \"" + prefix + "general-ai/Game-interfaces/TORCS/scr-client/classes;" + prefix + "general-ai/Game-interfaces/TORCS/scr-client/lib/*\""
TORCS_INSTALL_DIRECTORY_REF = prefix + "general-ai/Game-interfaces/TORCS/install_directory.txt"

# Config files for each game (contains I/O sizes)
GAME2048_CONFIG_FILE = prefix + "general-ai/Game-interfaces/Game2048/2048_config.json"
ALHAMBRA_CONFIG_FILE = prefix + "general-ai/Game-interfaces/Alhambra/Alhambra_config.json"
TORCS_CONFIG_FILE = prefix + "general-ai/Game-interfaces/TORCS/TORCS_config.json"
MARIO_CONFIG_FILE = prefix + "general-ai/Game-interfaces/Mario/Mario_config.json"
