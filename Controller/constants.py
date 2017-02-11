import os
import platform

loc = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
# prefix = Master directory
prefix = os.path.dirname(os.path.dirname(loc)) + "/"  # cut two last directories

delimiter = ";" if (platform.system() == "Windows") else ":"

MARIO = "java -cp \"" + prefix + "MarioAI/MarioAI4J/bin" + delimiter + prefix + "MarioAI/MarioAI4J-Playground/bin" + delimiter + prefix + "MarioAI/MarioAI4J-Playground/lib/*\" mario.GeneralAgent"
MARIO_VISUALISATION = "java -cp \"" + prefix + "MarioAI/MarioAI4J/bin" + delimiter + prefix + "MarioAI/MarioAI4J-Playground/bin" + delimiter + prefix + "MarioAI/MarioAI4J-Playground/lib/*\" mario.VisualizationTool"

GAME2048 = prefix + "2048/2048/bin/Release/2048.exe"
GAME2048_ADVANCED_TOOL = prefix + "2048/Visualization/bin/Release/Visualization.exe"

ALHAMBRA = prefix + "general-ai/Game-interfaces/Alhambra/AlhambraInterface/AlhambraInterface/bin/Release/AlhambraInterface.exe"

TORCS_BAT = "\"" + prefix + "general-ai/Game-interfaces/TORCS/torcs_starter.bat\""
TORCS_VIS_ON_BAT = "\"" + prefix + "general-ai/Game-interfaces/TORCS/torcs_starter_vis_on.bat\""
TORCS_SH = "\"" + prefix + "general-ai/Game-interfaces/TORCS/torcs_starter.sh\""
TORCS_VIS_ON_SH = "\"" + prefix + "general-ai/Game-interfaces/TORCS/torcs_starter_vis_on.sh\""
TORCS_XML = " \"" + prefix + "general-ai/Game-interfaces/TORCS/race_config.xml\""
TORCS_JAVA_CP = " \"" + prefix + "general-ai/Game-interfaces/TORCS/scr-client/classes;" + prefix + "general-ai/Game-interfaces/TORCS/scr-client/lib/*\""
TORCS_INSTALL_DIRECTORY_REF = prefix + "general-ai/Game-interfaces/TORCS/install_directory.txt"

# config files for each game (contains I/O sizes)
GAME2048_CONFIG_FILE = prefix + "general-ai/Game-interfaces/2048/2048_config.json"
ALHAMBRA_CONFIG_FILE = prefix + "general-ai/Game-interfaces/Alhambra/Alhambra_config.json"
TORCS_CONFIG_FILE = prefix + "general-ai/Game-interfaces/TORCS/TORCS_config.json"
MARIO_CONFIG_FILE = prefix + "general-ai/Game-interfaces/Mario/Mario_config.json"
