# Basic wrapper to start process with any game that has proper interface.
from __future__ import division
from __future__ import print_function

import random

import numpy as np

from evolution.differential_evolution import DifferentialEvolution
from evolution.evolution_parameters import EvolutionaryAlgorithmParameters, EvolutionStrategyParameters, \
    DifferentialEvolutionParameters
from evolution.evolution_strategy import EvolutionStrategy
from evolution.evolutionary_algorithm import EvolutionaryAlgorithm
from models.echo_state_network import EchoState
from models.mlp import MLP
from reinforcement.greedy_policy.greedy_policy_reinforcement import GreedyPolicyReinforcement
from reinforcement.ddpg.ddpg_reinforcement import DDPGReinforcement
from reinforcement.greedy_policy.q_network import QNetwork
from reinforcement.reinforcement_parameters import GreedyPolicyParameters, DDPGParameters

MASTER_SEED = 42
random.seed(MASTER_SEED)
np.random.seed(MASTER_SEED)


def run_eva():
    eva_parameters = EvolutionaryAlgorithmParameters(
        pop_size=15,
        cxpb=0.8,
        mut=("uniform", 0.1, 0.1),
        ngen=2000,
        game_batch_size=1,
        cxindpb=0.2,
        hof_size=0,
        elite=2,
        selection=("tournament", 3))

    mlp = MLP(hidden_layers=[256, 256], activation="relu")
    # esn = EchoState(n_readout=32, n_components=256, output_layers=[], activation="relu")
    evolution = EvolutionaryAlgorithm(game="torcs", evolution_params=eva_parameters, model=mlp, logs_every=10,
                                      max_workers=5)
    evolution.run()


def run_greedy():
    greedy_policy_params = GreedyPolicyParameters(
        batch_size=100,
        episodes=1000000,
        gamma=0.1,
        optimizer="adam",
        epsilon=0.1,
        test_size = 100,
        learning_rate=0.0001)

    q_net = QNetwork(hidden_layers=[256, 256, 256, 256, 256, 256], activation="relu", dropout_keep=None)
    RL = GreedyPolicyReinforcement(game="2048", parameters=greedy_policy_params, q_network=q_net, logs_every=20)
    RL.run()


def run_ddpg():
    ddpg_parameters = DDPGParameters(
        batch_size=100,
        episodes=1000000,
        test_size=100)

    RL = DDPGReinforcement(game="2048", parameters=ddpg_parameters, logs_every=100)
    RL.run()


def run_es():
    strategy_parameters = EvolutionStrategyParameters(
        pop_size=20,
        ngen=1000,
        game_batch_size=5,
        hof_size=0,
        elite=2,
        sigma=1.0)

    # mlp = MLP(hidden_layers=[16], activation="relu")
    esn = EchoState(n_readout=32, n_components=256, output_layers=[], activation="relu")
    strategy = EvolutionStrategy("alhambra", strategy_parameters, esn, logs_every=10, max_workers=5)
    strategy.run()


def run_de():
    diff_evolution_parameters = DifferentialEvolutionParameters(
        pop_size=20,
        ngen=1000,
        game_batch_size=5,
        hof_size=5,
        cr=0.25,
        f=1)

    mlp = MLP(hidden_layers=[32, 32], activation="relu")
    diff = DifferentialEvolution("alhambra", diff_evolution_parameters, mlp, max_workers=3, logs_every=10)
    diff.run()


if __name__ == '__main__':
    # run_greedy()
    run_ddpg()
    # run_es()
    # run_eva()
    # run_de()
