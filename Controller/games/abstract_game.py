import os


class AbstractGame():
    """ Basic wrapper for every game used."""

    def __init__(self):
        self.process = None
        self.model = None
        self.score = None
        self.score_extended = None

    def run(self, advanced_results=False):
        """
        Runs a whole game and returns result.
        :param advanced_results: If true, returns a list of results (if available).
        For example, returns scores of all players within the single game.
        :return: Game result.
        """
        state, current_phase = self.init_process()
        while True:
            result = self.model.evaluate(state, current_phase)

            state, current_phase, _, done = self.step(result)
            if done:
                if advanced_results:
                    return self.score_extended
                else:
                    return self.score

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
        phase = data["current_phase"]
        scores = data["score"]
        done = data["done"]
        self.score_extended = list(map(float, scores))
        self.score = self.score_extended[0]

        if int(done) == 1:
            self.finalize()
            return new_state, None, reward, True

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

    def send_to_process(self, input):
        """
        Sends the specified data to subprocess with the game.
        :param data: Data to be send.
        """
        data = ""
        for x in input:
            data += str(x) + " "
        data = "{}{}".format(data, os.linesep)
        self.process.stdin.write(bytearray(data.encode('ascii')))
        self.process.stdin.flush()

    def finalize(self, internal_error=False):
        """
        After game ends, do your stuff here (thread lock unlock for torcs...)
        :param internal_error: Determines whether the internal error occured.
        """
        self.process.kill()
