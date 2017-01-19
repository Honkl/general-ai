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
        super(Torcs, self).__init__()
        self.model = model
        self.game_batch_size = game_batch_size
        self.seed = seed

    def run(self, advanced_results=False):
        avg_result = 0
        for _ in range(self.game_batch_size):
            state, current_phase = self.init_process()
            while True:
                result = self.model.evaluate(state, current_phase)

                state, current_phase, _, done = self.step(result)
                if done:
                    avg_result += self.final_score[0]
                    break

        avg_result = avg_result / float(self.game_batch_size)
        return avg_result

    def init_process(self):
        Torcs.master_lock.acquire()

        index = np.random.randint(Torcs.MAX_NUMBER_OF_TORCS_PORTS)
        self.my_port_lock = Torcs.port_locks[index]
        for i in range(len(Torcs.port_locks)):
            if not Torcs.port_locks[i].locked():
                self.my_port_lock = Torcs.port_locks[i]
                index = i
                break
        self.my_port_lock.acquire()

        Torcs.master_lock.release()

        port_num = 3001 + index
        xml = " \"" + prefix + "general-ai\\Game-interfaces\\TORCS\\race_config_" + str(port_num) + ".xml\""
        port = " \"" + str(port_num) + "\""

        command = TORCS + xml + TORCS_JAVA_CP + port + TORCS_EXE_DIRECTORY
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        bufsize=-1)  # Using PIPEs is not the best solution...

        data = self.get_process_data()
        return data["state"], data["current_phase"]

    def get_process_data(self):
        line = ' '
        while line[0] != "{":
            # Not a proper json
            line = self.process.stdout.readline().decode('ascii')
        return json.loads(line)

    def finalize(self):
        self.my_port_lock.release()
