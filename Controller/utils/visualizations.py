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
from models.echo_state_network import EchoState


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

    ylim = get_y_lim_for_game(game)

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


def get_y_lim_for_game(game):
    ylim = None
    if game == "alhambra":
        ylim = 200
    if game == "torcs":
        ylim = 10000
    if game == "mario":
        ylim = 1.2
    if game == "2048":
        ylim = 5000
    return ylim


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


def compare_models(game, evals, *args):
    print("Comparing models:")
    values = []
    for model in args:
        print(model.get_name())
        values += eval(game=game, evals=evals, model=model)
    bar_plot(values, evals, game)


def eval_mario_winrate(model, evals, level, vis_on):
    game_instance = games.mario.Mario(model, evals, np.random.randint(0, 2 ** 16), level=level, vis_on=vis_on,
                                      use_visualization_tool=True)
    results = game_instance.run(advanced_results=True)
    print("Mario winrate: {}".format(results))


def run_torcs_vis_on(model, evals):
    game_instance = games.torcs.Torcs(model, evals, np.random.randint(0, 2 ** 16), vis_on=True)
    print("Torcs visualization started.")
    results = game_instance.run(advanced_results=True)


def run_2048_extended(model, evals):
    print("Game 2048 with extended logs started.")
    game_instance = games.game2048.Game2048(model, evals, np.random.randint(0, 2 ** 16), use_advanced_tool=True)
    results = game_instance.run(advanced_results=True)


def run_random_model(game, evals):
    print("Generating graph of 'random' model for game {}.".format(game))
    results = []
    for i in range(evals):
        print("{}/{}".format(i + 1, evals))
        parameters = [Random(game), 1, np.random.randint(0, 2 ** 16)]
        game_instance = utils.miscellaneous.get_game_instance(game, parameters)
        result = game_instance.run()
        results.append(result)

    x = range(evals)
    plt.plot(x, results, 'b', x, [np.mean(results) for _ in results], 'r--')
    plt.title("Random - game: {} - avg: {}".format(game, np.mean(results)))
    plt.ylim(0, get_y_lim_for_game(game))
    plt.savefig("random_model_{}.jpg".format(game))
    plt.show()


def eval_alhambra_winrate(model, evals):
    print("Evaluating Alhambra winrate.")
    results = []
    wins = 0
    for i in range(evals):
        print("{}/{}".format(i + 1, evals))
        game_instance = games.alhambra.Alhambra(model, 1, np.random.randint(0, 2 ** 16))
        result = game_instance.run(advanced_results=True)
        if np.argmax(result) == 0:
            wins += 1
    print("Alhambra winrate: {} ({}/{})".format(wins / evals, wins, evals))


if __name__ == '__main__':
    np.random.seed(930615)
    game = "alhambra"
    evals = 250

    # file_name = "../../Experiments/ESN+evolution_algorithm/2048/logs_2017-01-27_00-31-41/best/best_0.json"
    # file_name2 = "../../Controller/logs/2048/mlp/logs_2017-01-27_15-34-21/best/best_0.json"
    # file_name = "../../Experiments/ESN+evolution_algorithm/mario/logs_2017-01-28_16-10-43/best/best_0.json"
    # file_name = "../../Experiments/MLP+evolution_algorithm/alhambra/logs_2017-01-19_00-32-53/best/best_0.json"
    # file_name = "../../Experiments/ESN+evolution_algorithm/torcs/logs_2017-01-31_01-15-06/best/best_1.json"
    file_name = "../../Experiments/ESN+evolution_strategy/alhambra/logs_2017-01-30_13-24-16/best/best_1.json"

    esn = EchoState.load_from_file(file_name, game)
    # random = Random(game)
    # mlp = MLP.load_from_file(file_name, game)
    # eval_alhambra_winrate(esn, evals)
    # q_net = LearnedQNet(logdir)

    # run_random_model(game, evals)
    # run_2048_extended(mlp, evals)

    # eval_mario_winrate(model=esn, evals=evals, level="spikes", vis_on=False)
    compare_models(game, evals, esn)
    # run_torcs_vis_on(model=esn, evals=evals)
