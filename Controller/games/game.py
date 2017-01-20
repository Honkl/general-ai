import os


class Game():
    """ Basic wrapper for every game used."""

    def __init__(self):
        self.process = None
        self.model = None
        self.final_score = None

    def run(self, advanced_results=False):
        """
        Runs a whole game and returns result.
        :return:
        """
        state, current_phase = self.init_process()
        while True:
            result = self.model.evaluate(state, current_phase)

            state, current_phase, _, done = self.step(result)
            if done:
                if advanced_results:
                    return self.final_score
                else:
                    return self.final_score[0]

    def step(self, action):
        """
        Performs a single step within the game.
        :param action: Action to make.
        :return: New state, current phase, reward, done
        """
        self.send_to_process(action)
        data = self.get_process_data()

        reward = data["reward"]
        new_state = data["state"]
        if "final_score" in data:
            scores = data["final_score"]
            self.final_score = list(map(float, scores))
            self.finalize()
            return new_state, None, reward, True


        phase = data["current_phase"]
        return new_state, phase, reward, False

    def init_process(self):
        """
        Initializes a subprocess with the game and returns first state of the game.
        """
        raise NotImplementedError

    def get_process_data(self):
        """
        Gets a next data chunk from the game. Implementations are in child classes.
        :returns: Next game data.
        """
        raise NotImplementedError

    def send_to_process(self, data):
        """
        Sends the specified data to subprocess with the game.
        :param data: Data to be send.
        """
        data = "{}{}".format(data, os.linesep)
        self.process.stdin.write(bytearray(data.encode('ascii')))
        self.process.stdin.flush()

    def finalize(self):
        """
        After game ends, do your stuff here (thread lock unlock for torcs...)
        """
        pass
