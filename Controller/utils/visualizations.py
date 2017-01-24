import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import constants
import games
import utils.miscellaneous

from enum import Enum
from reinforcement.reinforcement import Reinforcement
from reinforcement.reinforcement_parameters import ReinforcementParameters
from reinforcement.q_network import QNetwork
from models.mlp import MLP
from models.learned_q_net import LearnedQNet
from models.random import Random


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


def compare(game, evals, *args):
    values = []
    for model in args:
        values += eval(game=game, evals=evals, model=model)
    bar_plot(values, evals, game)


def eval_mario_winrate(model, evals):
    game_instance = games.mario.Mario(model, evals, np.random.randint(0, 2 ** 16), level="gombas", vis_on=True,
                                      use_visualization_tool=True)
    results = game_instance.run(advanced_results=True)
    print("Mario winrate: {}".format(results))


def run_torcs_vis_on(model, evals):
    game_instance = games.torcs.Torcs(model, evals, np.random.randint(0, 2 ** 16), vis_on=True)
    print("Torcs visualization")
    results = game_instance.run(advanced_results=True)


if __name__ == '__main__':
    np.random.seed(42)
    game = "torcs"
    evals = 2

    # file_name = "../../Experiments/MLP+evolution_algorithm/2048/logs_2017-01-21_15-35-49/best/best_0.json"
    # file_name = "../../Experiments/MLP+evolution_algorithm/alhambra/logs_2017-01-19_00-32-53/best/best_1.json"
    file_name = "../../Experiments/MLP+evolution_algorithm/torcs/logs_2017-01-16_08-37-32/best/best_1.json"
    # file_name = "../../Experiments/MLP+evolution_algorithm/mario/logs_2017-01-22_00-46-06/best/best_0.json"
    # file_name = "../../Experiments/MLP+evolution_strategy/torcs/logs_2017-01-20_00-23-47/best/best_0.json"
    # logdir = "../../Controller/logs/2048/q-network/logs_2017-01-22_17-43-54"
    # file_name = "../../Controller/logs/2048/mlp/logs_2017-01-23_00-39-46/last/last_0.json"

    mlp = MLP.load_from_file(file_name, game)
    # q_net = LearnedQNet(logdir)
    random = Random(game)

    # eval_mario_winrate(model=mlp, evals=evals)
    # compare(game, evals, mlp, random)
    run_torcs_vis_on(model=mlp, evals=evals)