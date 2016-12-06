import os

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
# TORCS_EXE_DIRECTORY = " \"C:\\Users\\Jan\\Desktop\\torcs\""  # TODO: Relative path via cmd parameter
TORCS_EXE_DIRECTORY = " \"C:\\Program Files (x86)\\torcs\""  # TODO: Relative path via cmd parameter

# config files for each game (contains I/O sizes)
GAME2048_CONFIG_FILE = prefix + "general-ai\\Game-interfaces\\2048\\2048_config.json"
ALHAMBRA_CONFIG_FILE = prefix + "general-ai\\Game-interfaces\\Alhambra\\Alhambra_config.json"
TORCS_CONFIG_FILE = prefix + "general-ai\\Game-interfaces\\TORCS\\TORCS_config.json"
MARIO_CONFIG_FILE = prefix + "general-ai\\Game-interfaces\\Mario\\Mario_config.json"
