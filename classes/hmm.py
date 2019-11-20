import numpy as np
import math

abort_time = 2000  # Can be set to whatever feels right, after certain time remove object.
average_speed = 20  # The SCV moves in this speed, which is a good average for all units.


class HiddenMarkovModel:
    def __init__(self, columns, rows, start_time, map_area):
        self.start_time = start_time
        self.cell_size = map_area / (rows * columns)
        self.columns = columns
        self.rows = rows
        self.time_matrix = self.create_time_matrix()
        self.trans_matrix = np.full((rows, columns), 0.0)

    def on_step(self, log, time_frame):
        self.add_from_log(log)
        self.update_time_matrix(time_frame, log)
        print(self.trans_matrix)
        print(self.time_matrix)

    def get_most_likely(self):
        highest_prob = self.trans_matrix.max()
        indices = np.where(self.trans_matrix == highest_prob)  # change name
        position = (indices[1], indices[0])
        return highest_prob, position

    def create_time_matrix(self):
        time_matrix = []
        for i in range(self.rows):
            time_matrix.append([])
            for j in range(self.columns):
                time_matrix[i].append([])
        return time_matrix

    def add_from_log(self, log):
        print(log)
        for frame, positions in log.items():
            for position, units in log[frame].items():
                print(position)
                x_position = int(position[0])
                y_position = int(position[1])
                list_position = self.time_matrix[x_position][y_position]
                n_units_frame = (frame, len(units))
                if n_units_frame not in list_position:
                    list_position.append(n_units_frame)

    def update_time_matrix(self, current_frame, log):
        for i in range(self.rows):
            for j in range(self.columns):
                map_cell = self.time_matrix[i][j]
                if len(map_cell) > 0:
                    for n_units_frame in map_cell:
                        prob_units = self.calculate_probability_cell(i, j, n_units_frame[1])
                        if current_frame - n_units_frame[0] > abort_time:
                            print(log)
                            print(n_units_frame[0])
                            if n_units_frame[0] in log:
                                del log[n_units_frame[0]]
                            map_cell.remove(n_units_frame)
                            self.change_probability_trans_matrix(i, j, prob_units, n_units_frame[0], current_frame,
                                                                 self.remove_probability_trans_matrix)
                            # self.trans_matrix[self.columns - j - 1][i] = self.trans_matrix[self.columns - j - 1][i] - prob_units #CHANGE THIS
                        else:
                            self.change_probability_trans_matrix(i, j, prob_units, n_units_frame[0], current_frame,
                                                                 self.add_probability_trans_matrix)
                            # self.add_probabilily_trans_matrix(i, j, prob_units, n_units_frame[0], current_frame)

    def calculate_probability_cell(self, x_ratio, y_ratio, nr_units):
        possible_paths = 0
        # Count nr of cells
        for i in range(1, 3):
            for j in range(1, 3):
                if self.check_in_range(x_ratio + (i - 1), y_ratio + (j - 1)):
                    possible_paths += 1
        prob_unit = nr_units / possible_paths
        return prob_unit

    def change_probability_trans_matrix(self, x_cell_pos, y_cell_pos, prob_units, time_spotted, current_time,
                                        change_trans_matrix):
        print("SIZE OF CELL:  " + str(self.cell_size))
        print("TIME SPOTTED:  " + str(time_spotted))
        print("X:   " + str(x_cell_pos) + ",  Y:  " + str(y_cell_pos))

        steps = max(0, (current_time - time_spotted) // math.sqrt(self.cell_size))

        # Split to possibilities
        for i in range(1, 3):
            for j in range(1, 3):
                x = x_cell_pos + (i - 1)
                y = y_cell_pos + (j - 1)
                if self.check_in_range(x, y):
                    change_trans_matrix(x, y, prob_units)
                    print(steps)
                    if steps > 0:
                        new_time = time_spotted * (math.sqrt(self.cell_size) / average_speed)
                        prob_units = self.calculate_probability_cell(x, y, prob_units)
                        self.change_probability_trans_matrix(x, y, prob_units, new_time, current_time,
                                                             change_trans_matrix)

    def add_probability_trans_matrix(self, x, y, prob_units):
        self.trans_matrix[self.columns - 1 - y][x] = self.trans_matrix[self.columns - 1 - y][x] + prob_units

    def remove_probability_trans_matrix(self, x, y, prob_units):
        self.trans_matrix[self.columns - 1 - y][x] = self.trans_matrix[self.columns - 1 - y][x] - prob_units

    def check_in_range(self, i, j):
        if not (0 <= i <= self.rows - 1):
            return False
        elif not (0 <= j <= self.columns - 1):
            return False
        else:
            return True
