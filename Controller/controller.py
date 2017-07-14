"""
    GENERAL ARTIFICIAL INTELLIGENCE FOR GAME PLAYING
    JAN KLUJ, 2017.
    For more information on models or documentation how to use them, please visit:
    https://github.com/Honkl/general-ai
"""

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
from reinforcement.ddpg.ddpg_reinforcement import DDPGReinforcement
from reinforcement.reinforcement_parameters import DDPGParameters, DQNParameters
from reinforcement.dqn.dqn import DQN


# MASTER_SEED = 42
# random.seed(MASTER_SEED)
# np.random.seed(MASTER_SEED)


def run_eva(game):
    """
    EVOLUTIONARY ALGORITHM.
    """
    eva_parameters = EvolutionaryAlgorithmParameters(
        pop_size=25,
        cxpb=0.75,
        mut=("uniform", 0.1, 0.1),
        ngen=1000,
        game_batch_size=10,
        cxindpb=0.25,
        hof_size=0,
        elite=5,
        selection=("tournament", 3))

    # mlp = MLP(hidden_layers=[100, 100, 100, 100], activation="relu")
    esn = EchoState(n_readout=200, n_components=1000, output_layers=[], activation="relu")
    evolution = EvolutionaryAlgorithm(game=game, evolution_params=eva_parameters, model=esn, logs_every=100,
                                      max_workers=4)
    evolution.run()


def run_ddpg(game):
    """
    DEEP DETERMINISTIC POLICY GRADIENT (Reinforcement learning for games with continuous action spaces).
    """
    ddpg_parameters = DDPGParameters(
        batch_size=100,
        replay_buffer_size=100000,
        discount_factor=0.99,
        episodes=10000,
        test_size=25)

    print("DDPG algorithm started for game {}".format(game))
    print("Basic parameters: {}".format(ddpg_parameters.to_string()))

    # Parameters of networks are specified inside the DDPG Model. Using default parameters in most of the time.
    # For example, use path like this:
    # ckpt = "D:/general-ai-cache/logs/torcs/ddpg/logs_2017-07-01_21-40-57/"
    RL = DDPGReinforcement(game=game, parameters=ddpg_parameters, logs_every=50, checkpoint=None)
    RL.run()


def run_es(game):
    """
    EVOLUTIONARY STRATEGY (CMA-ES)
    """
    strategy_parameters = EvolutionStrategyParameters(
        pop_size=10,
        ngen=1000,
        game_batch_size=10,
        hof_size=0,
        elite=3,
        sigma=1.0)

    # mlp = MLP(hidden_layers=[50, 50, 50], activation="relu")
    esn = EchoState(n_readout=200, n_components=1000, output_layers=[], activation="relu")
    strategy = EvolutionStrategy(game, strategy_parameters, esn, logs_every=5, max_workers=4)
    strategy.run()


def run_de(game):
    """
    DIFFERENTIAL EVOLUTION
    """
    diff_evolution_parameters = DifferentialEvolutionParameters(
        pop_size=10,
        ngen=350,
        game_batch_size=1,
        hof_size=5,
        cr=0.25,
        f=1)

    # mlp = MLP(hidden_layers=[200, 200], activation="relu")
    esn = EchoState(n_readout=200, n_components=1000, output_layers=[], activation="relu")
    diff = DifferentialEvolution(game, diff_evolution_parameters, esn, max_workers=4, logs_every=5)
    diff.run()


def run_dqn(game):
    """
    DEEP REINFORCEMENT LEARNING (Q-network), epsilon greedy for exploration.
    """
    parameters = DQNParameters(batch_size=100,
                               init_exp=0.5,
                               final_exp=0.01,
                               anneal_steps=100000,
                               replay_buffer_size=100000,
                               store_replay_every=1,
                               discount_factor=0.99,
                               target_update_frequency=1000,
                               reg_param=0.01,
                               test_size=25)

    optimizer_params = {}
    optimizer_params["name"] = "adam"
    optimizer_params["learning_rate"] = 0.01

    q_network_parameters = {}
    q_network_parameters["hidden_layers"] = [500, 500]
    q_network_parameters["activation"] = "relu"
    q_network_parameters["dropout"] = 0.9

    RL = DQN(game, parameters, q_network_parameters, optimizer_params, test_every=50)
    RL.run()


if __name__ == '__main__':
    # Select the game: 2048, mario, torcs, alhambra
    game = "2048"

    # Select learning method
    run_eva(game)
    # run_es(game)
    # run_de(game)
    # run_dqn(game)
    # run_ddpg(game)
