from evolution.evolution import Evolution
from deap import tools, creator, base, cma
import time
import numpy as np
import concurrent.futures


class EvolutionStrategy(Evolution):
    def __init__(self, game, evolution_params, model, max_workers, logs_every=50):
        super(EvolutionStrategy, self).__init__(game, evolution_params, model, max_workers, logs_every)

    def run(self, file_name=None):
        """
        Starts evolution strategy (CMA-ES).
        :param file_name: Previously saved population file.
        """

        start_time = time.time()
        logs_dir = self.init_directories()

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        toolbox.register("map", executor.map)
        toolbox.register("evaluate", self.eval_fitness)

        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        N = self.model.get_number_of_parameters()
        print("N: {}".format(N))
        strategy = cma.Strategy(centroid=[0.0] * N, sigma=self.evolution_params.sigma,
                                lambda_=self.evolution_params.pop_size)
        print("CMA strategy created in {} sec".format(time.time() - start_time))
        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update", strategy.update)

        if (self.evolution_params.hof_size > 0):
            hof = tools.HallOfFame(self.evolution_params.hof_size)
        else:
            hof = None

        print("Evolution strategy started")
        for gen in range(1, self.evolution_params.ngen + 1):

            # Generate a new population
            if (gen == 1) and (file_name != None):
                population = self.init_population(pop_size=self.evolution_params.pop_size, container=list,
                                                  ind_init=toolbox.individual, file_name=file_name)
            else:
                population = toolbox.generate()

            # Evaluate the individuals
            seeds = [np.random.randint(0, 2 ** 16) for _ in range(len(population))]
            fitnesses = toolbox.map(toolbox.evaluate, population, seeds)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            if hof is not None:
                hof.update(population)

            st = time.time()
            # Update the strategy with the evaluated individuals
            toolbox.update(population)
            print("Population updated in {} sec".format(time.time() - st))

            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=gen, nevals=len(population), **record)
            print(logbook.stream)

            if (gen % self.logs_every == 0):
                self.log_all(logs_dir, population, hof, logbook, start_time)

        self.log_all(logs_dir, population, hof, logbook, start_time)