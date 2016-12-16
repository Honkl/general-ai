import matplotlib.pyplot as plt
import numpy as np
import constants

from games.alhambra import Alhambra
from games.torcs import Torcs
from games.mario import Mario
from games.game2048 import Game2048

EVALS = 100
np.random.seed(42)
GAME = "2048"

models = []
models.append(("Random", constants.loc + "/config/2048/random/random.json"))
models.append(("EVA + Feedforward", constants.loc + "/config/2048/feedforward/logs_2016-12-16_21-02-27/best_0.json"))


def plot_graph(values):
    n_groups = 1

    fig, ax = plt.subplots(figsize=(7, 7))
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = plt.bar(index, values[0][1], bar_width,
                     alpha=opacity,
                     color='b',
                     error_kw=error_config,
                     label=values[0][0])

    rects2 = plt.bar(index + bar_width, values[1][1], bar_width,
                     alpha=opacity,
                     color='r',
                     error_kw=error_config,
                     label=values[1][0])

    plt.ylim([0, 4000])
    plt.ylabel('AVG fitness')
    plt.title('Model comparison (based on {} runs)'.format(EVALS))
    plt.legend(loc='upper center', fancybox=True, shadow=True, ncol=5)

    autolabel(rects1, ax)
    autolabel(rects2, ax)

    plt.tight_layout()
    plt.savefig('comparison.jpg')
    plt.show()


def autolabel(rects, ax):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                '%d' % int(height),
                ha='center', va='bottom')


if __name__ == '__main__':

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

        values.append((name, game.run()))

    plot_graph(values)
