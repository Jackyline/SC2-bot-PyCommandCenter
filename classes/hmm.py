import numpy as np
import math
import munkres

abort_time = 2500  # Can be set to whatever feels right, after certain time remove object.
average_speed = 0.2  # The SCV moves in this speed, which is a good average for all units.


class HiddenMarkovModel:
    def __init__(self, columns, rows, start_time, map_area):
        self.start_time = start_time
        self.cell_size = map_area / (rows * columns)
        self.columns = columns
        self.rows = rows
        self.emission_matrix = self.create_emission_matrix()
        self.trans_matrix = np.full((rows, columns), 0.0)

    def on_step(self, log, time_frame):
        self.add_from_log(log)
        self.update_time_matrix(time_frame, log)
        # munkres.print_matrix(self.trans_matrix)

    def get_trans_matrix(self):
        """
        Used to print on the screen for probability in scouting manager
        :return: Transition matrix
        """
        return self.trans_matrix

    def get_most_likely(self):
        """
        Checks in the transition matrix were and what the highest probability is.
        :return: The highest probability and the points were it's located.
        """
        highest_prob = np.amax(self.trans_matrix)
        indices = np.where(self.trans_matrix == highest_prob)
        goals = list(zip(indices[1], indices[0]))
        return highest_prob, goals

    def create_emission_matrix(self):
        """
        Create the emission matrix, used when HMM is init
        :return: A emission matrix model
        """
        emission_matrix = []
        for i in range(self.columns):
            emission_matrix.append([])
            for j in range(self.rows):
                emission_matrix[i].append([])
        return emission_matrix

    def add_from_log(self, log):
        """
        Checks the log and appends new units spotted to the time matrix.
        :param log: A log which contains were and what units has been spotted
        """
        for frame, positions in log.items():
            for position, units in log[frame].items():
                x_position = int(position[:len(position)//2])
                y_position = int(position[len(position)//2:])
                if self.check_in_range(x_position, y_position):
                    list_position = self.emission_matrix[x_position][y_position]
                    # Added new line.
                    prob_units = self.calculate_probability_cell(x_position, y_position, len(units))

                    n_units_frame = (frame, prob_units)
                    if n_units_frame not in list_position:
                        list_position.append(n_units_frame)

    def update_time_matrix(self, current_frame, log):
        """
        Updates the time matrix, adds if new spotted to the transition matrix.
        If the time since spotted has past is greater than the abort time,
        then delete it from the log, time- and transition matrix
        :param current_frame: The current time frame
        :param log: A log which has all the enemy units spotted
        """
        for i in range(self.columns):
            for j in range(self.rows):
                map_cell = self.emission_matrix[i][j]
                if len(map_cell) > 0:
                    for k in range(len(map_cell) - 1, -1, -1):
                        n_units_frame = map_cell[k]
                        prob_units = self.calculate_probability_cell(i, j, n_units_frame[1])
                        if current_frame - n_units_frame[0] > abort_time:
                            if n_units_frame[0] in log:
                                del log[n_units_frame[0]]
                            del map_cell[k]
                            self.remove_probability_trans_matrix(i, j, n_units_frame[1], n_units_frame)
                        elif prob_units > 0.1:
                            self.fwd(i, j, prob_units, n_units_frame[0], current_frame,
                                     self.add_probability_trans_matrix)
                map_cell.sort(key=self.get_time)

    def get_time(self, elem):
        return elem[0]

    def calculate_probability_cell(self, x_ratio, y_ratio, nr_units):
        """
        Calculates the probability by dividing the number of units to the
        number of cells were the unit could have traveled
        :param x_ratio: The x-position for the map cell
        :param y_ratio: The y-position for the map cell
        :param nr_units: Number of units spotted
        :return: Probability of units
        """
        possible_paths = 0
        # Count nr of cells
        for i in range(1, 3):
            for j in range(1, 3):
                if self.check_in_range(x_ratio + (i - 1), y_ratio + (j - 1)):
                    possible_paths += 1
        # Check this later
        if possible_paths is 0:
            possible_paths = 1
        prob_unit = nr_units / possible_paths
        return prob_unit

    def fwd(self, x_cell_pos, y_cell_pos, prob_units, time_spotted, current_time,
            change_trans_matrix):
        """
        Change probability in the given cell and it neighbours depending on the time since it was spotted.
        :param x_cell_pos: The x-position of the map cell
        :param y_cell_pos: The y-position of the map cell
        :param prob_units: Probability of units being there
        :param time_spotted: Time frame when the units were spotted
        :param current_time: The current frame
        :param change_trans_matrix: Function which adds or deletes to the transition matrix
        """

        steps = max(0, (current_time - time_spotted) * average_speed // math.sqrt(self.cell_size))
        # Split to possibilities
        for i in range(1, 4):
            for j in range(1, 4):
                x = x_cell_pos + (i - 1)
                y = y_cell_pos + (j - 1)
                n_units_frame = (time_spotted, prob_units)
                if self.check_in_range(x, y):
                    change_trans_matrix(x, y, prob_units, n_units_frame)
                    if steps >= 1:
                        new_time = time_spotted + (math.sqrt(self.cell_size) / average_speed)
                        prob_units = self.calculate_probability_cell(x, y, prob_units)
                        if prob_units > 0.1:
                            self.fwd(x, y, prob_units, new_time, current_time,
                                     change_trans_matrix)

    def add_probability_trans_matrix(self, x, y, prob_units, n_units_frame):
        if prob_units > 0.1:
            if n_units_frame not in self.emission_matrix[x][y]:
                self.emission_matrix[x][y].append(n_units_frame)
                self.trans_matrix[y][x] = self.trans_matrix[y][x] + prob_units

    def remove_probability_trans_matrix(self, x, y, prob_units, n_units_frame):
        self.trans_matrix[y][x] = self.trans_matrix[y][x] - prob_units
        if self.trans_matrix[y][x] < 0.01:
            self.trans_matrix[y][x] = 0

    def check_in_range(self, i, j):
        """
        Checks if i and j is in valid range (Not out of bounds)
        :param i: The x-position of the map cell
        :param j: The y-position of the map cell
        :return: Boolean if i and j is in valid range
        """
        if not (2 <= i <= self.columns - 3):
            return False
        elif not (3 <= j <= self.rows - 3):
            return False
        else:
            return True
