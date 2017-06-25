import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, json

import games
import time
import utils.miscellaneous
from models.mlp import MLP
from models.echo_state_network import EchoState
from models.random import Random
from models.learned_dqn import LearnedDQN
from models.learned_ddpg import LearnedDDPG


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
    plt.savefig('comparison.png')


def get_y_lim_for_game(game):
    ylim = None
    if game == "alhambra":
        ylim = 200
    if game == "torcs":
        ylim = 3000
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

    game_instance = utils.miscellaneous.get_game_instance(game, parameters, test=True)

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
    """
    Evaluates mario winrate on specified level.
    :param model:
    :param evals:
    :param level: gombas or spikes
    :param vis_on:
    :return:
    """
    game_instance = games.mario.Mario(model, evals, np.random.randint(0, 2 ** 16), level=level, vis_on=vis_on,
                                      use_visualization_tool=True)
    results = game_instance.run(advanced_results=True)
    print("Mario winrate (avg dist): {}".format(results))


def run_torcs_vis_on(model, evals):
    game_instance = games.torcs.Torcs(model, evals, np.random.randint(0, 2 ** 16), vis_on=True)
    print("Torcs visualization started.")
    results = game_instance.run(advanced_results=True)


def run_2048_extended(model, evals):
    print("Game 2048 with extended logs started.")
    game_instance = games.game2048.Game2048(model, evals, np.random.randint(0, 2 ** 16))
    results = game_instance.run(advanced_results=True)
    return results


def run_random_model(game, evals):
    print("Generating graph of 'random' model for game {}.".format(game))
    results = []
    t = time.time()
    for i in range(evals):
        if time.time() - t > 1 or i == evals - 1:
            print("{}/{}".format(i + 1, evals))
            t = time.time()
        parameters = [Random(game), 1, np.random.randint(0, 2 ** 16)]
        game_instance = utils.miscellaneous.get_game_instance(game, parameters)
        result = game_instance.run()
        results.append(result)

    x = range(0, evals)
    # plt.plot(x, results, 'b', x, [np.mean(results) for _ in results], 'r--')
    plt.scatter(x, results, cmap='b')
    plt.plot([np.mean(results) for _ in results], 'r--')
    plt.title("Random - game: {} - Average score: {}".format(game, np.mean(results)))
    plt.ylim(0, get_y_lim_for_game(game))
    plt.xlim(0, evals)
    plt.xlabel("Evals")
    plt.ylabel("Score")
    plt.savefig("random_model_{}.png".format(game))


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


# INFERENCE METHOD
def run_model_evaluator(log_name):
    """
    Used for evaluating learned models, to benchmark them and get avg results.
    For example, to run 1000 games and plot results.
    Set file_name for example as "C:/Users/Jan/Documents/GitHub/general-ai/Experiments/ESN+EA/torcs/logs_2017-06-20_19-09-27/best/best_0.json" (the whole path)
    for evolutionary based experiments. For deep reinforcement (DQN or DDPG) based techniques use logdir, for example as:
    "C:/Users/Jan/Documents/GitHub/general-ai/Experiments/DQN/mario/logs_2017-04-19_16-26-07", where directory stores config files (model settings), model checkpoint etc.
    Then feel free to use some of the prepared "test functions". Result (if exists) is written to same directory as this file (e.q. /utils).
    """

    np.random.seed(930615)

    # Before using game 2048, check it's encoding
    game = "2048"
    evals = 1000

    # SELECT FILE (direct model for evolutionary or directory for reinforcement)
    file_name = "C:/Users/Jan/Documents/GitHub/general-ai/Experiments/MLP+DE/torcs/logs_2017-06-23_01-34-41/best/best_0.json"
    logdir = "C:/Users/Jan/Documents/GitHub/general-ai/Experiments/DDPG/torcs/logs_2017-05-01_14-37-32"

    # SELECT MODEL (trained, based on file selected)
    esn = EchoState.load_from_file(file_name, game)
    # mlp = MLP.load_from_file(file_name, game)
    # random = Random(game)
    # ddpg = LearnedDDPG(logdir)
    # dqn = LearnedDQN(logdir)

    # RUN MODEL TEST
    # eval_alhambra_winrate(mlp, evals)
    # run_random_model(game, evals)
    run_2048_extended(esn, evals)
    # eval_mario_winrate(model=esn, evals=evals, level="spikes", vis_on=False)
    # run_torcs_vis_on(model=mlp, evals=evals)

    # general model comparison (graph of score)
    # compare_models(game, evals, esn)

    """
    NOTE: Selected file source file, selected model (python object) and the game must be correct (must match). If you save model for
    game 2048 using ESN, you can't load this model as DDPG for TORCS of course.
    """


def run_avg_results():
    items = ["logs_2017-06-23_14-16-00",
             "logs_2017-06-23_14-16-59",
             "logs_2017-06-23_14-17-58",
             "logs_2017-06-23_14-18-48",
             "logs_2017-06-23_14-19-39"]

    results = []
    game = "2048"
    evals = 1000
    for item in items:
        prefix = "C:/Users/Jan/Documents/GitHub/general-ai/Experiments/best_models_repeats/2048/MLP+ES/"
        postfix = "/best/best_0.json"
        file_name = prefix + item + postfix

        model = MLP.load_from_file(file_name, game)
        result = run_2048_extended(model, evals)
        results.append(result)

    results = np.array(results)
    file_name = "game2048_stats_{}.txt".format(utils.miscellaneous.get_pretty_time())
    with open(file_name, "w") as f:
        f.write("--GAME 2048 STATISTICS-- {} trainings of same model".format(len(items)))
        f.write(os.linesep)
        f.write("Model: {}".format(model.get_name()))
        f.write(os.linesep)
        f.write("Total games: {}".format(evals))
        f.write(os.linesep)
        f.write("MAX TEST: {}".format(np.max(results)))
        f.write(os.linesep)
        f.write("AVG TEST: {}".format(np.mean(results)))
        f.write(os.linesep)
        f.write("MIN TEST: {}".format(np.min(results)))

# GRAPH CREATOR
def run_plot_creator():
    """
    Used for creating some graphs, plots, etc...
    For evolution-based experiments only. For deep reinforcement learning experiments, please use proper TensorBoard.
    """

    # Set directory of model (example: C:/Users/Jan/Documents/GitHub/general-ai/Experiments/ESN+DE/mario/logs_2017-05-04_23-08-42):
    dir_name = "C:/Users/Jan/Documents/GitHub/general-ai/Experiments/ESN+EA/torcs/logs_2017-06-20_19-09-27"
    game = "TORCS"

    with open(os.path.join(dir_name, "settings.json"), "r") as f:
        metadata = json.load(f)

    data = np.loadtxt(os.path.join(dir_name, "logbook.txt"), skiprows=1)
    episodes = data[:, 0]
    scores = data[:, 2]

    plt.figure()
    plt.plot(episodes, scores, label="avg fitness in generation")
    i = np.argmax(scores)
    plt.scatter(i, scores[i])
    plt.text(i, scores[i], "{}".format(round(max(scores), 2)))

    # Plot the graph, for different game, use different settings
    params = "EVA + ESN"
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.xlim([0, len(episodes)])
    plt.ylim([0, 1000])
    plt.legend(loc="lower right")
    plt.title("GAME: {}\n{}".format(game, params, fontsize=10))
    plt.savefig("plot.pdf")


if __name__ == '__main__':
    # INFERENCE OF MODELS FUNCTION
    # run_model_evaluator()
    run_avg_results()

    # RESULT GRAPH GENERATOR
    # run_plot_creator()
