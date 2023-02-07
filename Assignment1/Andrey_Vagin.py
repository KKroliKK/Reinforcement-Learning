import itertools
import numpy as np
from typing import Dict, Tuple


def read_map(filename):
    map = []

    with open(filename) as file:
        text = file.read()
    lines = text.splitlines()

    for line in lines:
        line = [int(cell) for cell in line.split()]
        map.append(line)

    return map




class Cargo:

    def __init__(self, map) -> None:
        self.map = map
        self.shape = self.get_cargo_shape()

    
    def get_right_lower_corner(self, cargo):
        lower = 0
        right = 0

        for row, col in cargo:
            lower = max(lower, row)
            right = max(right, col)
        
        return lower, right


    def get_cargo_shape(self):
        '''Returns list of tupples with shift of cargo's cells
        with respect to right lower corner (if it even does not exist).
        '''
        cargo = []

        for row in range(len(self.map)):
            for col in range(len(self.map[0])):
                if self.map[row][col] == 2:
                    cargo.append((row, col))

        self.corner_row, self.corner_col = self.get_right_lower_corner(cargo)

        for part_idx in range(len(cargo)):
            row, col = cargo[part_idx]
            cargo[part_idx] = (row - self.corner_row, col - self.corner_col)
        
        return cargo

    
    def get_cargo_coordinates(self, corner_row, corner_col):
        cargo_coordinates = []

        for row_shift, col_shift in self.shape:
            cargo_coordinates.append((corner_row + row_shift, corner_col + col_shift))

        return cargo_coordinates

    
    def is_valid_position(self, corner_row, corner_col):

        cargo_coordinates = self.get_cargo_coordinates(corner_row, corner_col)

        def is_valid_cell(row, col):
            n_rows = len(self.map)
            n_cols = len(self.map[0])

            if row < 0:
                return False
            if row >= n_rows:
                return False
            if col < 0:
                return False
            if col >= n_cols:
                return False

            if self.map[row][col] == 1:
                return False

            return True

        check = [is_valid_cell(row, col) for row, col in cargo_coordinates]
        is_valid = all(check)

        return is_valid




class GridWorld:

    class Cell:

        def __init__(self, row, col, world, default_reward=-1):

            self.row = row
            self.col = col
            self.env = world
            self.default_reward = default_reward

            self.actions_to = []

        def compute_value(self, gamma=0.9):
            if len(self.actions_to) == 0:
                return self.env.value_function[str(self)]
            qs = []
            for place in self.actions_to:
                action_reward = self.env.rewards.get((str(self), place), self.default_reward)
                next_v = gamma * self.env.value_function[place]
                qs.append(action_reward + next_v)
            return max(qs)

        def __str__(self):
            return f"{self.row}_{self.col}"

        def __repr__(self):
            return str(self)


    def __init__(self,
                 map: list,
                 rewards: Dict[Tuple[str, str], float] = {}, 
                 gamma: float = 0.9,
                 default_reward: float = -1):
                 
        self.map = map
        self.n_rows = len(map)
        self.n_cols = len(map[0])
        
        coords = itertools.product(range(self.n_rows), range(self.n_cols))
        
        self.state_list = [self.Cell(row, col, self, default_reward) for row, col in coords]
        self.state_dict = {str(cell): cell for cell in self.state_list}
        self.value_function = {str(cell): 0 for cell in self.state_list}
        self.rewards = rewards
        self.gamma = gamma


    def update_values(self):
        new_value_function = self.value_function.copy()
        
        for cell in self.state_list:
            new_value_function[str(cell)] = cell.compute_value(self.gamma)
        
        self.value_function = new_value_function


    def visualize(self):
        array = np.zeros((self.n_rows, self.n_cols))
        for col in range(self.n_cols):
            for row in range(self.n_rows):
                array[row, col] = self.value_function[f'{row}_{col}']

        print(array)

    def get_state_value(self):
        array = np.zeros((self.n_rows, self.n_cols))
        for col in range(self.n_cols):
            for row in range(self.n_rows):
                array[row, col] = self.value_function[f'{row}_{col}']

        return array




def get_state_value_of_map(map):
    cargo = Cargo(map)
    world = GridWorld(map)

    for row in range(world.n_rows):
        for col in range(world.n_cols):

            if cargo.is_valid_position(row, col) == False:
                world.value_function[f'{row}_{col}'] = -np.inf
                continue

            moves = [(1, 0), (-1, 0), (0, -1), (0, 1)]

            for row_shift, col_shift in moves:
                new_row = row + row_shift
                new_col = col + col_shift

                if cargo.is_valid_position(new_row, new_col):
                    world.state_dict[f'{row}_{col}'].actions_to.append(f'{new_row}_{new_col}')

    world.state_dict[f'{world.n_rows - 1}_{world.n_cols - 1}'].actions_to = []

    for _ in range(100):
        world.update_values()
    
    return world.get_state_value()




class StateValueMap:

    def __init__(self, state_value_map):
        self.map = state_value_map
        self.n_rows = len(state_value_map)
        self.n_cols = len(state_value_map[0])

    def is_valid_position(self, row, col):
        if row < 0:
            return False
        if row >= self.n_rows:
            return False
        if col < 0:
            return False
        if col >= self.n_cols:
            return False

        if self.map[row][col] == -np.inf:
            return False

        return True

    def get_possible_moves(self, row, col):
        positions = []

        moves = [(1, 0), (-1, 0), (0, -1), (0, 1)]

        for row_shift, col_shift in moves:
            new_row = row + row_shift
            new_col = col + col_shift

            if self.is_valid_position(new_row, new_col):
                positions.append((new_row, new_col))

        return positions

    def get_best_moves(self, row, col):
        positions = self.get_possible_moves(row, col)
        maximum = -np.inf

        for row, col in positions:
            maximum = max(maximum, self.map[row][col])

        moves = []
        for row, col in positions:
            if self.map[row][col] == maximum:
                moves.append((row, col))

        return moves




def get_path_array(map):

    state_map = StateValueMap(get_state_value_of_map(map))
    cargo = Cargo(map)

    came_from = [len(map[0]) * [None] for _ in range(len(map))]

    queue = [(cargo.corner_row, cargo.corner_col)]

    while len(queue) != 0:
        cell = queue.pop(0)
        row, col = cell

        moves = state_map.get_best_moves(row, col)

        for move in moves: 
            row, col = move
            if came_from[row][col] == None:
                came_from[row][col] = cell
                queue.append(move)

    came_from[cargo.corner_row][cargo.corner_col] = cargo.corner_row, cargo.corner_col

    return came_from




def get_path(map):
    came_from = get_path_array(map)

    if came_from[-1][-1] == None:
        return []

    path = [came_from[-1][-1]]
    
    row, col = came_from[-1][-1]

    while came_from[row][col] != (row, col):
        row, col = came_from[row][col]
        path.append((row, col))

    path.reverse()
    path.append((len(map) - 1, len(map[0]) - 1))

    return path




def get_steps(map):
    path = get_path(map)

    if len(path) == 0:
        return 'No path'

    steps = ''

    for i in range(len(path) - 1):
        row_from, col_from = path[i]
        row_to, col_to = path[i + 1]

        if (row_to - row_from) == -1:
            steps += 'U '

        elif (row_to - row_from) == 1:
            steps += 'D '

        elif (col_to - col_from) == -1:
            steps += 'L '
        
        elif (col_to - col_from) == 1:
            steps += 'R '

    return steps




def find_path(path_to_infile, path_to_out_file):
    map = read_map(path_to_infile)
    path = get_steps(map)

    with open(path_to_out_file, 'w') as file:
        file.write(path)



# for i in range(6):
#     find_path(f'./inputs/input{i}.txt', f'./outputs/output{i}.txt')