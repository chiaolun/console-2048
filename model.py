import copy
import random
import functools


def push_row(row, left=True):
    """Push all tiles in one row; like tiles will be merged together."""
    row = row[:] if left else row[::-1]
    new_row = [item for item in row if item]
    for i in range(len(new_row)-1):
        if new_row[i] and new_row[i] == new_row[i+1]:
            new_row[i], new_row[i+1:] = new_row[i]*2, new_row[i+2:]+[0]
    new_row += [0]*(len(row)-len(new_row))
    return new_row if left else new_row[::-1]


def get_column(grid, column_index):
    """Return the column from the grid at column_index  as a list."""
    return [row[column_index] for row in grid]


def set_column(grid, column_index, new):
    """
    Replace the values in the grid at column_index with the values in new.
    The grid is changed inplace.
    """
    for i, row in enumerate(grid):
        row[column_index] = new[i]


def push_all_rows(grid, left=True):
    """
    Perform a horizontal shift on all rows.
    Pass left=True for left and left=False for right.
    The grid will be changed inplace.
    """
    for i, row in enumerate(grid):
        grid[i] = push_row(row, left)


def push_all_columns(grid, up=True):
    """
    Perform a vertical shift on all columns.
    Pass up=True for up and up=False for down.
    The grid will be changed inplace.
    """
    for i, val in enumerate(grid[0]):
        column = get_column(grid, i)
        new = push_row(column, up)
        set_column(grid, i, new)

move_list = [functools.partial(push_all_rows, left=True),
             functools.partial(push_all_rows, left=False),
             functools.partial(push_all_columns, up=True),
             functools.partial(push_all_columns, up=False)]


def get_empty_cells(grid):
    """Return a list of coordinate pairs corresponding to empty cells."""
    empty = []
    for j, row in enumerate(grid):
        for i, val in enumerate(row):
            if not val:
                empty.append((j, i))
    return empty


def any_possible_moves(grid):
    """Return True if there are any legal moves, and False otherwise."""
    if get_empty_cells(grid):
        return True
    for row in grid:
        if any(row[i] == row[i+1] for i in range(len(row)-1)):
            return True
    for i, val in enumerate(grid[0]):
        column = get_column(grid, i)
        if any(column[i] == column[i+1] for i in range(len(column)-1)):
            return True
    return False


def get_start_grid(cols=4, rows=4):
    """Create the start grid and seed it with two numbers."""
    grid = [[0]*cols for i in range(rows)]
    for i in range(2):
        empties = get_empty_cells(grid)
        y, x = random.choice(empties)
        grid[y][x] = 2 if random.random() < 0.9 else 4
    return grid


def prepare_next_turn(grid):
    """
    Spawn a new number on the grid; then return the result of
    any_possible_moves after this change has been made.
    """
    empties = get_empty_cells(grid)
    y, x = random.choice(empties)
    grid[y][x] = 2 if random.random() < 0.9 else 4
    return any_possible_moves(grid)


def print_grid(grid):
    """Print a pretty grid to the screen."""
    print("")
    wall = "+------"*len(grid[0])+"+"
    print(wall)
    for row in grid:
        meat = "|".join("{:^6}".format(val) for val in row)
        print("|{}|".format(meat))
        print(wall)


def score_grid(grid):
    return max([i for row in grid for i in row])


class Game:
    def __init__(self, cols=4, rows=4):
        self.grid = get_start_grid(cols, rows)
        self.score = score_grid(self.grid)
        self.end = False

    def post_states(self):
        for move in move_list:
            grid_copy = copy.deepcopy(self.grid)
            move(grid_copy)
            if grid_copy != self.grid:
                yield grid_copy
            else:
                yield None

    def move(self, direction):
        grid_copy = copy.deepcopy(self.grid)
        move_list[direction](self.grid)

        if self.grid == grid_copy:
            return None

        if prepare_next_turn(self.grid):
            score = score_grid(self.grid)
            reward = score - self.score
            self.score = score
            return reward

        self.end = True
        return 0

    def display(self):
        print_grid(self.grid)
