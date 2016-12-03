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

    def __init__(self, model_config_file, game_batch_size):
        self.model_config_file = model_config_file
        self.game_batch_size = game_batch_size

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

        avg_result = 0
        for _ in range(self.game_batch_size):

            MCF = " \"" + self.model_config_file + "\""
            command = TORCS + xml + TORCS_JAVA_CP + port + PYTHON_SCRIPT + TORCS_EXE_DIRECTORY + PYTHON_EXE + MCF
            p = subprocess.Popen(command, stdout=subprocess.PIPE)
            result = p.communicate()[0].decode('ascii')
            result = re.split("\\r\\n|\\n", result)
            distances = []
            for line in result:
                if "RACED DISTANCE:" in line:
                    distances.append(line.split(":")[1].strip())
            avg_result += float(distances[0])

        avg_result = avg_result / float(self.game_batch_size)
        my_port_lock.release()
        return avg_result
