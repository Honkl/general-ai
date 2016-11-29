from games.game import Game
import subprocess
import re
import numpy as np
from threading import Lock
from constants import *


class Torcs(Game):
    MAX_NUMBER_OF_TORCS_PORTS = 10

    master_lock = Lock()
    port_locks = [Lock() for _ in range(MAX_NUMBER_OF_TORCS_PORTS)]

    def __init__(self, model_config_file):
        self.model_config_file = model_config_file

    def run(self):

        Torcs.master_lock.acquire()

        index = np.random.randint(Torcs.MAX_NUMBER_OF_TORCS_PORTS)
        my_port_lock = Torcs.port_locks[index]
        for i in range(len(Torcs.port_locks)):
            if not Torcs.port_locks[i].locked():
                my_port_lock = Torcs.port_locks[i]
                index = i
                break
        my_port_lock.acquire()

        Torcs.master_lock.release()

        port_num = 3001 + index
        xml = " \"" + prefix + "general-ai\\Game-interfaces\\TORCS\\race_config_" + str(port_num) + ".xml\""
        port = " \"" + str(port_num) + "\""

        MCF = " \"" + self.model_config_file + "\""
        command = TORCS + xml + TORCS_JAVA_CP + port + PYTHON_SCRIPT + TORCS_EXE_DIRECTORY + PYTHON_EXE + MCF
        p = subprocess.Popen(command, stdout=subprocess.PIPE)
        result = p.communicate()[0].decode('ascii')

        my_port_lock.release()

        result = re.split("\\r\\n|\\n", result)
        distances = []
        print(result)
        for line in result:
            if "RACED DISTANCE:" in line:
                distances.append(line.split(":")[1].strip())
        return float(distances[0])
