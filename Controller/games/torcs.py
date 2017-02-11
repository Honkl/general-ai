from games.game import Game
import subprocess
import json
import numpy as np
from threading import Lock
from constants import *
import platform


class Torcs(Game):
    MAX_NUMBER_OF_TORCS_PORTS = 10

    master_lock = Lock()
    port_locks = [Lock() for _ in range(MAX_NUMBER_OF_TORCS_PORTS)]

    def __init__(self, model, game_batch_size, seed, vis_on=False):
        """
        Initializes a new instance of TORCS game.
        :param model: Model which will be playing this game.
        :param game_batch_size: Number of games that will be played immediately (one after one) within the single game
        instance. Result is averaged.
        :param seed: A random seed for random generator within the game.
        :param vis_on: Determines whether TORCS will run with visual output. If True, different subprocess will be used.
        """
        super(Torcs, self).__init__()
        self.model = model
        self.game_batch_size = game_batch_size
        self.seed = seed
        self.vis_on = vis_on

    def run(self, advanced_results=False):
        """
        Starts a single TORCS game.
        :return: TORCS game result (passed distance for example).
        """
        avg_result = 0
        for _ in range(self.game_batch_size):
            state, current_phase = self.init_process()
            while True:
                result = self.model.evaluate(state, current_phase)

                state, current_phase, _, done = self.step(result)
                if done:
                    avg_result += self.score[0]
                    break

        avg_result = avg_result / float(self.game_batch_size)
        if advanced_results:
            return [avg_result]
        return avg_result

    def init_process(self):
        """
        Initializes a new process with TORCS game. Maximum 10 process at time (maximum 10 ports that are available for
        TORCS).
        """
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

        with open(TORCS_INSTALL_DIRECTORY_REF, "r") as f:
            torcs_install_dir = f.readline()

        windows = platform.system() == "Windows"
        if self.vis_on:
            if windows:
                params = [TORCS_VIS_ON_BAT, xml, TORCS_JAVA_CP, port, torcs_install_dir]
                command = "{} {} {} {} {}".format(*params)
            else:
                params = [TORCS_VIS_ON_SH, xml, TORCS_JAVA_CP, port, torcs_install_dir]
                command = ["sh"] + params
        else:
            if windows:
                params = [TORCS_BAT, xml, TORCS_JAVA_CP, port, torcs_install_dir]
                command = "{} {} {} {} {}".format(*params)
            else:
                params = [TORCS_SH, xml, TORCS_JAVA_CP, port, torcs_install_dir]
                command = ["sh"] + params

        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=-1)

        data = self.get_process_data()
        return data["state"], data["current_phase"]

    def get_process_data(self):
        """
        Gets a subprocess next data (line).
        :return: a subprocess next data (line).
        """
        line = ' '
        while line[0] != "{":
            # Not a proper json
            line = self.process.stdout.readline().decode('ascii')

        return json.loads(line)

    def finalize(self):
        """
        Finalizes the game subprocess. Releases used locks and kills the subprocess.
        :return:
        """
        try:
            self.my_port_lock.release()
        except:
            pass

        self.process.kill()
