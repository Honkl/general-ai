# Basic wrapper to start process with any game that has proper interface.
from __future__ import print_function
from __future__ import division

import time
import numpy as np
import matplotlib.pyplot as plt
import random

from deap import tools
from evolution.evolution import Evolution
from evolution.evolutionary_algorithm import EvolutionaryAlgorithm
from evolution.evolution_strategy import EvolutionStrategy
from evolution.differential_evolution import DifferentialEvolution
from evolution.evolution_parameters import EvolutionaryAlgorithmParameters, EvolutionStrategyParameters, \
    DifferentialEvolutionParameters
from models.mlp import MLP
from models.echo_state_network import EchoState

from utils.activations import relu, tanh, logsig
from reinforcement.reinforcement import Reinforcement
from reinforcement.reinforcement_parameters import ReinforcementParameters
from reinforcement.q_network import QNetwork, QNetworkRnn

MASTER_SEED = 42
random.seed(MASTER_SEED)
np.random.seed(MASTER_SEED)

sea_params = EvolutionaryAlgorithmParameters(
    pop_size=15,
    cxpb=0.75,
    mut=("uniform", 0.1, 0.1),
    ngen=500,
    game_batch_size=1,
    cxindpb=0.2,
    hof_size=0,
    elite=2,
    selection=("tournament", 3))

es_params = EvolutionStrategyParameters(
    pop_size=20,
    ngen=1000,
    game_batch_size=5,
    hof_size=0,
    elite=2,
    sigma=1.0)

rl_params = ReinforcementParameters(
    batch_size=1,
    epochs=10000,
    gamma=0.7,
    optimizer="adam",
    rand_action_prob=0.9)

de_params = DifferentialEvolutionParameters(
    pop_size=25,
    ngen=5000,
    game_batch_size=100,
    hof_size=5,
    elite=0,
    cr=0.25,
    f=1)


def run_eva():
    # mlp = MLP(hidden_layers=[256, 256], activation="relu")
    esn = EchoState(n_readout=16, n_components=128, output_layers=[], activation="relu")
    evolution = EvolutionaryAlgorithm(game="torcs", evolution_params=sea_params, model=esn, logs_every=10,
                                      max_workers=3)
    evolution.run()


def run_reinforcement():
    print(rl_params.to_string())
    # q_net = QNetwork(hidden_layers=[256, 256], activation="relu", dropout_keep=0.5)
    q_net = QNetworkRnn(rnn_cell_type="lstm", num_units=256)
    RL = Reinforcement("2048", rl_params, q_net, threads=10)
    RL.run()


def run_es():
    # mlp = MLP(hidden_layers=[16], activation="relu")
    esn = EchoState(n_readout=32, n_components=256, output_layers=[], activation="relu")
    strategy = EvolutionStrategy("alhambra", es_params, esn, logs_every=10, max_workers=5)
    strategy.run()


def run_de():
    mlp = MLP(hidden_layers=[64, 64], activation="relu")
    diff = DifferentialEvolution("2048", de_params, mlp, max_workers=25, logs_every=10)
    diff.run()


if __name__ == '__main__':
    run_reinforcement()
    # run_es()
    # run_eva()
    # run_de()
