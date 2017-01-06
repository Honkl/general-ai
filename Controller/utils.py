import matplotlib.pyplot as plt
import numpy as np
import constants

from games.alhambra import Alhambra
from games.torcs import Torcs
from games.mario import Mario
from games.game2048 import Game2048


def plot_graph(values):
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

    plt.ylim([0, 200])
    plt.gca().axes.set_xticklabels([])
    plt.ylabel('AVG fitness')
    plt.title('Model comparison (based on {} runs)'.format(EVALS))

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


if __name__ == '__main__':

    np.random.seed(42)
    EVALS = 100
    GAME = "alhambra"

    models = []
    #models.append(("Random", constants.loc + "/config/alhambra/random/random.json"))
    models.append(("EVA + Feedforward FC", constants.loc + "/config/alhambra/feedforward/logs_2017-01-03_11-12-59/best_0.json"))

    values = []
    for model in models:
        name = model[0]
        model_config_file = model[1]

        params = [model_config_file, EVALS, np.random.randint(0, 2 ** 16)]

        game = None
        if GAME == "alhambra":
            game = Alhambra(*params)
        if GAME == "2048":
            game = Game2048(*params)
        if GAME == "mario":
            game = Mario(*params)
        if GAME == "torcs":
            game = Torcs(*params)

        game_result = game.run(advanced_results=True)
        values.append((name, game_result[0]))
        values.append(("Original#1", game_result[1]))
        values.append(("Original#2", game_result[2]))

    plot_graph(values)
