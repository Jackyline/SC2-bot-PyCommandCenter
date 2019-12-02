import numpy as np
from munkres import print_matrix


class Hungarian():
    """
    Implementation of the Hungarian algorithm for the maximization assignment problem.
    Used for assigning tasks to units in StarCraft.
    """

    def __init__(self):
        """
        Initialize instance of the assignment problem with the profit matrix
        """
        self.is_balanced = True
        self.original_matrix_rows = 0
        self.original_matrix_cols = 0
        self.n = 0
        self.V = self.X = self.Y = 0
        self.matrix = None
        self.x_labels = None
        self.y_labels = None
        self.matching = None
        self.inverse_matching = None
        self.total_profit = None
        self.minSlack = None

    def generate_matrix(self, utility_func):
        """
        Generates a profit matrix based on a utility function that is used to calculate the profit of assigning each task to each units. Profit matrix is made balanced but nr of dummy row/cols are saved to be removed later
        :param utility_func: Function that takes parameters :unit and :task and returns a profit (int) of assigning a task to a unit
        :return: NxN profit matrix. Profit from unit X assigned to task Y.
        """

        n_units = len(self.units)
        n_tasks = len(self.tasks)
        max_n = max(n_units, n_tasks)

        # Save nr dummy row/cols to remove later
        self.original_matrix_rows = n_units
        self.original_matrix_cols = n_tasks
        self.is_balanced = (n_units == n_tasks)

        self.matrix = np.zeros((max_n, max_n))  # This way balanced matrix is balanced right away
        for i in range(len(self.units)):
            for j in range(len(self.tasks)):
                self.matrix[i][j] = utility_func(self.units[i], self.tasks[j])

        if self.debug:
            print_matrix(self.matrix, msg='\nmatrix generated from utility function')

    def balance_matrix(self):
        """
        Pads 0 values to rectangular matrices to make them squared,
        making turning an unbalanced assignment problemn into a balanced problem.
        """

        # Save number of dummy row/cols to be removed from computed assignments
        (a, b) = self.matrix.shape
        self.original_matrix_rows = a
        self.original_matrix_cols = b
        self.is_balanced = (a == b)

        if not self.is_balanced:
            (a, b) = self.matrix.shape
            if a > b:
                padding = ((0, 0), (0, a - b))
            else:
                padding = ((0, b - a), (0, 0))
            self.matrix = np.pad(self.matrix, padding, mode='constant', constant_values=0)

    def compute_assignments(self, matrix):
        """
        Initialize new problem instance by resetting variables, and creating a profit matrix
        and then compute maximum matching from X to Y.
        :param matrix: NxN profit matrix. Profit from X assigned to Y.
        :return: maximum matching from X to Y. dict<int, int>
        """
        assert len(matrix) != 0

        self.matrix = matrix
        self.balance_matrix()
        self.n = len(self.matrix)
        self.V = self.X = self.Y = set(range(
            self.n))  # using set<int> of interval (0, n) to represent X and Y. V is used when creating for loops on sets other than X or Y but of equal size/representation.
        self.init_labels()
        self.matching = {}  # The matching is adict with node in X as key with matching node in Y as value
        self.inverse_matching = {}
        self.find_augmented_path_and_augment()
        self.remove_dummy_assignments()
        self.total_profit = sum(self.matrix[x][y] for x, y in self.matching.items())
        return self.matching

    def remove_dummy_assignments(self):
        """
        Remove any assignments in matching that are involve any dummy row or column
        """
        if not self.is_balanced:
            if self.original_matrix_rows > self.original_matrix_cols:
                self.matching = {k: v for k, v in self.matching.items() if (
                            v < self.original_matrix_cols or self.original_matrix_rows < v)}  # Create copy of matching where all keys with dummy values have been removed

            else:
                for i in range(self.original_matrix_rows, self.original_matrix_cols):
                    del self.matching[i]  # Remove dummy keys

    def init_labels(self):
        """
        Initialize the labelling for each node x to the max weight of all it's edges.
        Initialize the labelling for each node y to 0.
        """
        self.x_labels = [0 for _ in self.X]
        self.y_labels = [0 for _ in self.Y]
        for x in self.X:
            for y in self.Y:
                self.x_labels[x] = max(self.x_labels[x], self.matrix[x][y])

    def slack(self, x, y):
        """
        Calculate delta to use when updating slack values
        """
        return self.x_labels[x] + self.y_labels[y] - self.matrix[x][y]

    def find_augmented_path_and_augment(self):
        """
        Main function of the algorithm. Find augmenting path and augment the current
        matching until a perfect matching is found.
        """
        while True:

            if len(self.matching) == self.n:  # Has found a perfect matching
                return

            for x in self.X:  # Find an unmatched node in X and set as root
                if x not in self.matching:
                    root = x
                    break

            self.minSlack = [[self.slack(root, y), root] for y in self.Y]

            path = {root: None}
            S = set([root])
            T = set()

            x, y, path = self.find_augmenting_path(path, S, T)
            self.augment_matching(x, y, path)

    def find_augmenting_path(self, path, S, T):
        """
        Find an augmenting path for the current matching. If an augmenting path cannot be found the feasible labelling
        will be improved instead in order to expand the equality graph. This is done using the slack method.
        :param path: Traceable path to the root of the augmenting path.
                     Keys are nodes in X, values are the node in X preceding the key in the path to the root.
                     dict<int, int>
        :param S: set of nodes from X in the alternating tree. set<int>
        :param T: set of nodes from Y in the alternating tree. set<int>
        :return: a tuple (x, y, path) where (x, y) is the ending edge of the augmenting path and path is as above.
                 tuple(int, int, dict<int, int>)
        """

        while (True):
            # select edge (x,y) with x in S, y not in T and min slack
            ((val, x), y) = min([(self.minSlack[y], y) for y in self.Y if y not in T])
            assert x in S
            if val > 0:
                self.improve_labels(val, S, T)
            assert self.slack(x, y) == 0  # now the found y is part of equality graph, which means slack = 0

            if y in self.inverse_matching:  # y is matched -> Extend the alternating tree
                z = self.inverse_matching[y]
                assert not z in S
                S.add(z)
                T.add(y)
                path[z] = x

                for y in self.Y:  # Update slack
                    if not y in T and self.minSlack[y][0] > self.slack(z, y):
                        self.minSlack[y] = [self.slack(z, y), z]

            else:  # y is unmatched -> Augmenting path found
                return x, y, path

    def augment_matching(self, x, y, path):
        """
        Augments the current matching using the path ending with edge (x, y).
        (x, y) is not in the current matching. Neither is the edge to root.
        :param x: last node in X in the augmenting path to the root. int
        :param y: very end of the augmenting path. int
        :param path: Traceable path to the root of the augmenting path.
                     Keys are nodes in X, values are the nodes in X preceding the key in the path to the root.
                     dict<int, int>
        """

        if path[x] is None:
            # Root has been reached
            self.matching[x] = y
            self.inverse_matching[y] = x
            return

        # Swap x to be matched with y
        matched_y = self.matching[x]
        self.matching[x] = y
        self.inverse_matching[y] = x
        self.augment_matching(path[x], matched_y, path)

    def is_in_equality_graph(self, x, y):
        """
        Check if edge (x, y) is in the equality graph.
        :param x: node from X. int
        :param y: node from Y. int
        :return: True if (x, y) is in the equality graph.
        """
        return self.matrix[x][y] == self.x_labels[x] + self.y_labels[y]

    def improve_labels(self, val, S, T):
        """
        Improce the current labelling so that:
            - the current matching remains in the new equality graph
            - the current alternating path remains in the new equality graph
            - there is a free vertex from Y and not in T in the new equality graph
        When this is run the neighbourhood of S in the equality graph is equal to T.
        :param S: nodes from X in the alternating tree. set<int>
        :param T: nodes from Y in the alternating tree. set<int>
        """

        for v in self.V:
            if v in S:
                self.x_labels[v] -= val
            if v in T:
                self.y_labels[v] += val
            else:
                self.minSlack[v][0] -= val

    def pretty_print_assignments(self):
        """
        Pretty prints assignments and it's profit
        :param assignments: assignments from node in X to node in Y. dict<int, int>
        :param matrix: NxN profit matrix. Profit from X assigned to Y.
        :return: Total profit of assignments
        """

        print("\nAssignments: ")
        total_profit = 0
        for key, value in self.matching.items():
            profit = self.matrix[key][value]
            total_profit += profit
            print('(%d, %d) -> %d' % (key, value, profit))

        print('\nTotal profit: %d\n' % total_profit)
