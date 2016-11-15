class Game():
    """ Basic wrapper for every game used."""
    def run(self):
        raise NotImplementedError("NotImplementedException")

    def get_execution_time(self):
        raise NotImplementedError("NotImplementedException")