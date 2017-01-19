from games.game import Game
import subprocess
import json
import numpy as np
from threading import Lock
from constants import *


class Torcs(Game):
    MAX_NUMBER_OF_TORCS_PORTS = 10

    master_lock = Lock()
    port_locks = [Lock() for _ in range(MAX_NUMBER_OF_TORCS_PORTS)]

    def __init__(self, model, game_batch_size, seed):
        self.model = model
        self.game_batch_size = game_batch_size
        self.seed = seed

    def run(self, advanced_results=False):

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
            command = TORCS + xml + TORCS_JAVA_CP + port + TORCS_EXE_DIRECTORY
            p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 bufsize=-1)  # Using PIPEs is not the best solution...
            score = None
            while (True):
                line = p.stdout.readline().decode('ascii')

                if "RACED DISTANCE" in line:
                    score = line.split(":")[1].strip()
                    avg_result += float(score)
                    break

                if line[0] != "{":
                    # Not a proper json
                    continue

                #print("LINE: {}".format(line))
                result = self.model.evaluate(json.loads(line))
                result = "{}{}".format(result, os.linesep)

                p.stdin.write(bytearray(result.encode('ascii')))
                p.stdin.flush()



        avg_result = avg_result / float(self.game_batch_size)
        my_port_lock.release()
        return avg_result
