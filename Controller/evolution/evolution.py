from __future__ import print_function
from __future__ import division

import os
import json

import numpy as np
import concurrent.futures

import constants
import time
import matplotlib.pyplot as plt

from deap import creator, base, tools

from games.alhambra import Alhambra
from games.torcs import Torcs
from games.mario import Mario
from games.game2048 import Game2048


class Evolution():
    all_time_best = []

    def __init__(self, game, evolution_params, model, max_workers, logs_every=50):
        self.current_game = game
        self.evolution_params = evolution_params
        self.model = model
        self.max_workers = max_workers
        self.logs_every = logs_every

        game_config_file = ""
        if game == "alhambra":
            game_config_file = constants.ALHAMBRA_CONFIG_FILE
        if game == "2048":
            game_config_file = constants.GAME2048_CONFIG_FILE
        if game == "mario":
            game_config_file = constants.MARIO_CONFIG_FILE
        if game == "torcs":
            game_config_file = constants.TORCS_CONFIG_FILE

        with open(game_config_file, "r") as f:
            self.game_config = json.load(f)

    def write_to_file(self, individual, filename):
        """
        Writes individual to file for logging purposes.
        :param individual: Individual to log.
        :param filename: Filename where to write.
        """
        with open(filename, "w") as f:
            data = {}
            data["model_name"] = self.model.get_name()
            data["class_name"] = self.model.get_class_name()
            data["hidden_sizes"] = self.model.hidden_layers
            data["weights"] = individual
            data["activation"] = self.model.activation
            f.write(json.dumps(data))

    def eval_fitness(self, individual, seed):
        """
        Evaluates a fitness of the specified individual.
        :param individual: Individual whose fitness will be evaluated.
        :param seed: Seed for the game instance.
        :return: Fitness of the individual (must be tuple for Deap library).
        """
        # Need to create new instance of model (using specified weights). Also good usage for multi threading.
        model = self.model.get_new_instance(weights=individual, game_config=self.game_config)
        params = [model, self.evolution_params._game_batch_size, seed]

        game = None
        if self.current_game == "alhambra":
            game = Alhambra(*params)
        if self.current_game == "2048":
            game = Game2048(*params)
        if self.current_game == "mario":
            game = Mario(*params)
        if self.current_game == "torcs":
            game = Torcs(*params)

        result = game.run()

        return result,

    def mut_random(self, individual, mutindpb):
        """
        Provides random mutation of a individual.
        :param individual: Individual to mutate.
        :param mutindpb: Probability of mutation for single "bit" of individual.
        :return: New mutated individual.
        """
        for i in range(len(individual)):
            if (np.random.random() < mutindpb):
                individual[i] = np.random.random()
        return individual,

    def init_individual(self, icls, length, content=None):
        if content == None:
            return icls([np.random.random() for _ in range(length)])
        return icls(content)

    def init_population(self, pop_size, container, ind_init, file_name=None):
        if file_name == None:
            return container(ind_init() for _ in range(pop_size))

        with open(file_name) as f:
            content = json.load(f)
            return container(ind_init(content=x) for x in content["population"])

    def deap_toolbox_init(self):
        """
        Initializes the current instance of evolution.
        :returns: Deap toolbox.
        """
        individual_len = self.model.get_number_of_parameters(self.current_game)

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", np.random.random)
        toolbox.register("individual", self.init_individual, length=individual_len, icls=creator.Individual)
        toolbox.register("population", self.init_population, container=list, ind_init=toolbox.individual)

        toolbox.register("evaluate", self.eval_fitness)
        toolbox.register("mate", tools.cxUniform, indpb=self.evolution_params.cxindpb)

        mut_name = self.evolution_params.mut[0]
        if mut_name == "uniform":
            toolbox.register("mutate", self.mut_random, mutindpb=self.evolution_params.mut[2])
        else:
            raise NotImplementedError
            # toolbox.register("mutate", tools.mutGaussian, mu=0.5, sigma=0.05, indpb=0.05)

        sel = self.evolution_params.selection[0]
        if sel == "tournament":
            toolbox.register("select", tools.selTournament, tournsize=self.evolution_params.selection[1])
        elif sel == "selbest":
            toolbox.register("select", tools.selBest)
        else:
            raise NotImplementedError

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        toolbox.register("map", executor.map)
        return toolbox

    def create_log_files(self, dir, pop, log, elapsed_time):
        if not os.path.exists(dir):
            os.makedirs(dir)

        with open((dir + "\\pop.json"), "w") as f:
            data = {}
            data["population"] = pop
            f.write(json.dumps(data))

        with open((dir + "\\logbook.txt"), "w") as f:
            f.write(str(log))

        with open((dir + "\\settings.json"), "w") as f:
            data = {}
            data["evolution_params"] = self.evolution_params.to_dictionary()
            data["model_params"] = self.model.to_dictionary()
            f.write(json.dumps(data))

        with open((dir + "\\runtime.txt"), "w") as f:
            f.write("{}".format(elapsed_time))

        gen, avg, min_, max_ = log.select("gen", "avg", "min", "max")
        plt.figure()
        plt.plot(gen, avg, label="avg")
        plt.plot(gen, min_, label="min")
        plt.plot(gen, max_, label="max")
        i = np.argmax(avg)
        plt.scatter(gen[i], avg[i])
        plt.text(gen[i], avg[i], "{}".format(round(max(avg), 2)))
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.xlim([0, len(gen)])
        plt.legend(loc="lower right")
        plt.title("GAME: {}\n{}\n{}".format(self.current_game, self.evolution_params.to_string(),
                                            self.model.to_string()), fontsize=10)
        plt.savefig(dir + "\\plot.jpg")

    def init_directories(self):
        self.dir = constants.loc + "\\logs\\" + self.current_game + "\\" + self.model.get_name()
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        # create name for directory to store logs
        current = time.localtime()
        t_string = "{}-{}-{}_{}-{}-{}".format(str(current.tm_year).zfill(2),
                                              str(current.tm_mon).zfill(2),
                                              str(current.tm_mday).zfill(2),
                                              str(current.tm_hour).zfill(2),
                                              str(current.tm_min).zfill(2),
                                              str(current.tm_sec).zfill(2))

        return self.dir + "\\logs_" + t_string

    def log_all(self, logs_dir, population, hof, logbook, start_time):
        """
        Creates all logs of the current state of evolution.
        :param logs_dir: Logging directory.
        :param population: Population to log.
        :param hof: Hall of fame to log (can be None).
        :param logbook: Logbook info (from deap lib).
        :param start_time: Start time of evolution.
        """
        t = time.time() - start_time
        h = t // 3600
        m = (t % 3600) // 60
        s = t - (h * 3600) - (m * 60)
        elapsed_time = "{}h {}m {}s".format(int(h), int(m), s)

        self.create_log_files(logs_dir, population, logbook, elapsed_time)
        print("Time elapsed: {}".format(elapsed_time))

        best_dir = logs_dir + "/best"
        last_dir = logs_dir + "/last"

        if not os.path.exists(best_dir):
            os.makedirs(best_dir)
        if not os.path.exists(last_dir):
            os.makedirs(last_dir)

        number_to_log = max(self.evolution_params.hof_size, self.evolution_params.elite)
        for i in range(number_to_log):
            self.write_to_file(population[i], last_dir + "\\last_" + str(i) + ".json")
            self.all_time_best.append(population[i])

        self.all_time_best.sort(key=lambda ind: ind.fitness.values, reverse=True)
        self.all_time_best = self.all_time_best[:number_to_log]

        for i in range(number_to_log):
            self.write_to_file(self.all_time_best[i], best_dir + "\\best_" + str(i) + ".json")

    def run(self):
        raise NotImplementedError("Evolution class called. Maybe you should call one of the child classes.")
