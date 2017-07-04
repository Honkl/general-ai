"""
This is a test of Monte Carlo method for the Game 2048. This is not an 'official' part of "General artificial
intelligence for game playing" project, because it's applied only for this game. This was made only for test and
comparison purposes, out of curiosity, just to see how MC will perform on this task.
"""

import numpy as np
import time
import os
from game_2048 import Game
from multiprocessing import Pool

np.random.seed(42)

THREADS = 8
ITERS_PER_STEP = 100
GAMES_TO_PLAY = 1000


def monte_carlo(game_index):
    print("Starting game: {}".format(game_index))
    game = Game(seed=np.random.randint(low=0, high=2 ** 30))
    while not game.end:
        action = get_best_move(game)
        moved, _ = game.move(action)
    return game


def get_best_move(game):
    results = [0, 0, 0, 0]
    moves = [0, 1, 2, 3]
    for action in moves:
        game_copy = game.copy()
        moved, _ = game_copy.move(action)
        if moved:
            for iter in range(ITERS_PER_STEP):
                g = game_copy.copy()
                results[action] += random_play(g)

    for i in range(len(results)):
        results[i] /= ITERS_PER_STEP
    return np.argmax(results)


def random_play(game):
    moves = [0, 1, 2, 3]
    while not game.end:
        np.random.shuffle(moves)
        for m in moves:
            if game.move(m):
                break
    return game.score


def get_elapsed_time(start):
    now = time.time()
    t = now - start
    h = t // 3600
    m = (t % 3600) // 60
    s = t - (h * 3600) - (m * 60)
    elapsed_time = "{}h {}m {}s".format(int(h), int(m), s)
    return elapsed_time


if __name__ == '__main__':
    start = time.time()
    counts = {}
    results = []
    scores = []

    # Evaluate games
    p = Pool(THREADS)
    results = p.map(monte_carlo, range(GAMES_TO_PLAY))

    # Just logging stuff and print results
    for i in range(GAMES_TO_PLAY):
        completed_game = results[i]
        scores.append(completed_game.score)
        m = completed_game.max()
        if m in counts:
            counts[m] += 1
        else:
            counts[m] = 1
        print("Iteration: {}: Score: {}, Max: {}".format(i + 1, completed_game.score, completed_game.max()))

    end = time.time()

    print(counts)
    file_name = "game2048_MC_depth_{}x{}.txt".format(GAMES_TO_PLAY, ITERS_PER_STEP)
    with open(file_name, "w") as f:
        f.write("--GAME 2048 MONTE CARLO STATISTICS--")
        f.write(os.linesep)
        f.write("Model: Monte Carlo (MC) [only for 2048 out of curiosity purposes]")
        f.write(os.linesep)
        f.write("Total Runtime: {}, Avg time per game: {}sec".format(get_elapsed_time(start),
                                                                     (end - start) / GAMES_TO_PLAY))
        f.write(os.linesep)
        f.write("Total Games: {}, Average score: {}".format(GAMES_TO_PLAY, np.mean(scores)))
        f.write(os.linesep)
        f.write("Reached Tiles:")
        f.write(os.linesep)

        width = 5
        for key in sorted(counts):
            f.write("{}: {} = {}%".format(str(key).rjust(width), str(counts[key]).rjust(width),
                                          str(100 * counts[key] / GAMES_TO_PLAY).rjust(width)))
            f.write(os.linesep)
    print("Total time: {}".format(get_elapsed_time(start)))
