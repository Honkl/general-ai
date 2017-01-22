import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from enum import Enum
from reinforcement.reinforcement import Reinforcement
from reinforcement.reinforcement_parameters import ReinforcementParameters
from reinforcement.q_network import QNetwork
from models.mlp import MLP
from models.random import Random
import utils.miscellaneous


def bar_plot(values, evals, game):
    fig, ax = plt.subplots(figsize=(6, 7))

    bar_width = 0.3
    opacity = 0.5

    for index, (name, value) in enumerate(values):
        rects = plt.bar(index, value, bar_width,
                        alpha=opacity,
                        color='b',
                        label=name,
                        align='center')
        autolabel(rects, ax)

    ylim = None
    if game == "alhambra":
        ylim = 200
    if game == "torcs":
        ylim = 10000
    if game == "mario":
        ylim = 1.2
    if game == "2048":
        ylim = 5000

    plt.ylim([0, ylim])
    plt.gca().axes.set_xticklabels([])
    plt.ylabel('AVG fitness')
    plt.title('Model comparison - {} runs - {}'.format(evals, game))

    x = np.arange(len(values))
    ax.set_xticks(x)
    ax.set_xticklabels([name for (name, _) in values])

    plt.tight_layout()
    plt.savefig('comparison.jpg')
    plt.show()


def autolabel(rects, ax):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                '{}'.format(float(height)),
                ha='center', va='bottom')


def eval(game, evals, model):
    parameters = [model, evals, np.random.randint(0, 2 ** 16)]
    values = []
    game_instance = utils.miscellaneous.get_game_instance(game, parameters)

    results = game_instance.run(advanced_results=True)
    for i, r in enumerate(results):
        if i > 0:
            values.append(("original#{}".format(i), r))
        else:
            values.append((model.get_name(), r))
    return values


if __name__ == '__main__':
    np.random.seed(42)
    evals = 1

    game = "mario"

    # file_name = "../../Experiments/MLP+evolution_algorithm/2048/logs_2017-01-21_15-35-49/best/best_0.json"
    # file_name = "../../Experiments/MLP+evolution_algorithm/alhambra/logs_2017-01-19_00-32-53/best/best_1.json"
    # file_name = "../../Experiments/MLP+evolution_algorithm/torcs/logs_2017-01-16_08-37-32/best/best_0.json"
    file_name = "../../Experiments/MLP+evolution_algorithm/mario/logs_2017-01-22_00-46-06/best/best_0.json"
    # file_name = "../../Experiments/MLP+evolution_strategy/torcs/logs_2017-01-20_00-23-47/best/best_0.json"

    mlp = MLP.load_from_file(file_name, game)
    random = Random(game)

    values = []
    values += eval(game=game, evals=evals, model=random)
    values += eval(game=game, evals=evals, model=mlp)
    bar_plot(values, evals, game)
