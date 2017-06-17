# modified version of https://github.com/Mekire/console-2048/blob/master/console2048.py
# and https://github.com/tjwei/2048-NN/blob/master/c2048.py
# some small modifications were made (reward).

import numpy as np


def push_left(grid):
    moved, score = False, 0
    rows, columns = grid.shape[0], grid.shape[1]
    for k in range(rows):
        i, last = 0, 0
        for j in range(columns):
            e = grid[k, j]
            if e:
                if e == last:
                    grid[k, i - 1] += e
                    score += e
                    last, moved = 0, True
                else:
                    moved |= (i != j)
                    last = grid[k, i] = e
                    i += 1
        while i < columns:
            grid[k, i] = 0
            i += 1
    return score if moved else -1


def push_right(grid):
    moved, score = False, 0
    rows, columns = grid.shape[0], grid.shape[1]
    for k in range(rows):
        i = columns - 1
        last = 0
        for j in range(columns - 1, -1, -1):
            e = grid[k, j]
            if e:
                if e == last:
                    grid[k, i + 1] += e
                    score += e
                    last, moved = 0, True
                else:
                    moved |= (i != j)
                    last = grid[k, i] = e
                    i -= 1
        while 0 <= i:
            grid[k, i] = 0
            i -= 1
    return score if moved else -1


def push_up(grid):
    moved, score = False, 0
    rows, columns = grid.shape[0], grid.shape[1]
    for k in range(columns):
        i, last = 0, 0
        for j in range(rows):
            e = grid[j, k]
            if e:
                if e == last:
                    score += e
                    grid[i - 1, k] += e
                    last, moved = 0, True
                else:
                    moved |= (i != j)
                    last = grid[i, k] = e
                    i += 1
        while i < rows:
            grid[i, k] = 0
            i += 1
    return score if moved else -1


def push_down(grid):
    moved, score = False, 0
    rows, columns = grid.shape[0], grid.shape[1]
    for k in range(columns):
        i, last = rows - 1, 0
        for j in range(rows - 1, -1, -1):
            e = grid[j, k]
            if e:
                if e == last:
                    score += e
                    grid[i + 1, k] += e
                    last, moved = 0, True
                else:
                    moved |= (i != j)
                    last = grid[i, k] = e
                    i -= 1
        while 0 <= i:
            grid[i, k] = 0
            i -= 1
    return score if moved else -1


def push(grid, direction):
    if direction & 1:
        if direction & 2:
            score = push_down(grid)
        else:
            score = push_up(grid)
    else:
        if direction & 2:
            score = push_right(grid)
        else:
            score = push_left(grid)
    return score


def put_new_cell(grid, rng):
    n = 0
    r = 0
    i_s = [0] * 16
    j_s = [0] * 16
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not grid[i, j]:
                i_s[n] = i
                j_s[n] = j
                n += 1
    if n > 0:
        r = rng.randint(0, n)
        grid[i_s[r], j_s[r]] = 2 if rng.random_sample() < 0.9 else 4
    return n


def any_possible_moves(grid):
    """Return True if there are any legal moves, and False otherwise."""
    rows = grid.shape[0]
    columns = grid.shape[1]
    for i in range(rows):
        for j in range(columns):
            e = grid[i, j]
            if not e:
                return True
            if j and e == grid[i, j - 1]:
                return True
            if i and e == grid[i - 1, j]:
                return True
    return False


def prepare_next_turn(grid, rng):
    """
    Spawn a new number on the grid; then return the result of
    any_possible_moves after this change has been made.
    """
    empties = put_new_cell(grid, rng)
    return empties > 1 or any_possible_moves(grid)


def print_grid(grid_array):
    """Print a pretty grid to the screen."""
    print("")
    wall = "+------" * grid_array.shape[1] + "+"
    print(wall)
    for i in range(grid_array.shape[0]):
        meat = "|".join("{:^6}".format(grid_array[i, j]) for j in range(grid_array.shape[1]))
        print("|{}|".format(meat))
        print(wall)


class Game:
    def __init__(self, seed, cols=4, rows=4):
        self.cols = cols
        self.rows = rows
        self.rng = np.random.RandomState(seed)
        self.grid_array = np.zeros(shape=(rows, cols), dtype='uint16')
        self.grid = self.grid_array
        for i in range(2):
            put_new_cell(self.grid, self.rng)
        self.score = 0
        self.end = False
        self.total_moves = 0

    def copy(self):
        rtn = Game(self.grid.shape[0], self.grid.shape[1])
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                rtn.grid[i, j] = self.grid[i, j]
        rtn.score = self.score
        rtn.end = self.end
        return rtn

    def max(self):
        m = 0
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i, j] > m:
                    m = self.grid[i, j]
        return m

    def move(self, direction):
        if direction & 1:
            if direction & 2:
                score = push_down(self.grid)
            else:
                score = push_up(self.grid)
        else:
            if direction & 2:
                score = push_right(self.grid)
            else:
                score = push_left(self.grid)
        if score == -1:
            return 0, None
        score *= 2  # We want result as a score (2 + 2 merged should be score "4" not "2")
        reward = score
        self.total_moves += 1
        self.score += score
        if not prepare_next_turn(self.grid, self.rng):
            self.end = True
        return 1, reward

    def display(self):
        print_grid(self.grid_array)

    def get_state(self):
        # FEEL FREE CHANGE THIS ENCODING

        # return self.get_state_onehot()
        return self.get_state_raw()

    def get_state_raw(self):
        return np.array([np.log2(x) if x > 0 else .0 for x in self.grid.flatten()])

    def get_state_onehot(self):
        MAX_POWER = 16
        table = {2 ** (i + 1): i for i in range(MAX_POWER)}
        x = np.zeros(shape=(self.rows, self.cols, MAX_POWER), dtype=float)
        for i in range(self.rows):
            for j in range(self.cols):
                value = self.grid[i, j]
                if value > 0:
                    x[i, j][table[value]] = 1
        return x.flatten()


""" # Debug stuff
g = Game(seed=0)
g.display()
g.move(1)
g.display()
print("score: {}, max: {}".format(g.score, g.max()))
"""

"""
def random_play(game):
    moves = [0, 1, 2, 3]
    while not game.end:
        shuffle(moves)
        for m in moves:
            if game.move(m):
                break
    return game.score
"""
