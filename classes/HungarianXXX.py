from munkres import Munkres, print_matrix
import sys
import numpy as np
from random import randint

class Hungarian(object):
    """
    Implementation of the Hungarian algorithm for the maximization assignment problem.
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

    def balance_matrix(self):
        """ Pads 0 values to rectangular matrices to make them square """

        (a,b) = self.matrix.shape
        self.original_matrix_rows = a
        self.original_matrix_cols = b
        self.is_balanced = (a == b)

        if not self.is_balanced:
            (a,b) = self.matrix.shape
            if a>b:
                padding=((0,0),(0,a-b))
            else:
                padding=((0,b-a),(0,0))
            self.matrix = np.pad(self.matrix,padding,mode='constant',constant_values=0)

    def compute_assignments(self, matrix):
        """
        Compute maximum matching from X to Y. dict<int, int>
        :param matrix: NxN profit matrix. Profit from X assigned to Y.
        :return: maximum matching from X to Y. dict<int, int>
        """

        # TODO: Handle floats
        # TODO: Dummy stuff
        # TODO: Set n below
        # TODO. Work jobs enter where
        # TODO: Calc matrix from jobs and workds
        # TODO: create profit matrix with help of utility function, set n to nr of jobs/workers

        self.matrix = matrix
        self.balance_matrix()
        self.n = len(self.matrix)
        self.V = self.X = self.Y = set(range(self.n))  # using set<int> of interval (0, n) to represent X and Y. V is used when creating for loops on sets other than X or Y but of equal size/representation.
        self.init_labels()
        self.matching = {}  # The matching is adict with node in X as key with matching node in Y as value
        self.inverse_matching = {}
        self.find_augmented_path_and_augment()
        self.remove_dummy_assignments()
        self.total_profit = sum(self.matrix[x][y] for x, y in self.matching.items())
        return self.matching

    def remove_dummy_assignments(self):
        """ Remove any assignments in matching that are involve any dummy row or column """
        pass

    def pretty_print(self):
        for n in range(len(self.matching)):
            self.matching[n] = self.jobs[self.matching[n]]
            self.matching[self.workers[n]] = self.matching.pop(n)

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
        """ calculate TODO """
        return self.x_labels[x] + self.y_labels[y] - self.matrix[x][y]

    def find_augmented_path_and_augment(self):
        """
        Core of the algorithm. Find augmenting path and augment the current
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

        while(True):
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

                for y in self.Y: # Update slack
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

class TestHungarian():
    """ Imported Munkres module that provides an implementation of the Munkres algorithm
        that can be used to test my implementation. Generates 10 balanced matrices and 10
        unbalanced matrices of maximum size 'max_problem_size that tests are run on '"""

    def __init__(self, max_problem_size):
        self.h = Hungarian()
        self.m = Munkres()
        self.matrices = []
        self.rectangular_matrices = []

        # Generate random matrices used for testing
        for i in range(1, 11):
            rand_int1 = randint(1, max_problem_size)
            rand_int2 = randint(1, max_problem_size)
            self.matrices.append(np.random.randint(100, size=(rand_int1, rand_int1)))  # balanced matrices
            self.rectangular_matrices.append(np.random.randint(100, size=(rand_int1, rand_int2)))  # unbalanced matrices

    def max_to_min_problem(self, matrix):
        """ Convert from maximization to min problem """
        cost_matrix = []
        for row in matrix:
            cost_row = []
            for col in row:
                cost_row += [sys.maxsize - col]
            cost_matrix += [cost_row]
        return cost_matrix

    def convert_to_dict(self, indexes):
        """ This module uses lists to represents assignments whereas my
        implementation uses dict """
        assignments = {}
        for pair in indexes:
            assignments[pair[0]] = pair[1]
        return assignments

    def compute_test_assignments(self, matrix):
        """ Compute assignments using imported munkres """
        cost_matrix = self.max_to_min_problem(matrix)
        indexes = self.m.compute(cost_matrix)
        total = 0
        return self.convert_to_dict(indexes)

    def print_balanced_matrix(self, matrix, msg):
        (a,b) = matrix.shape
        if a>b:
            padding=((0,0),(0,a-b))
        else:
            padding=((0,b-a),(0,0))
        balanced_matrix = np.pad(matrix,padding,mode='constant',constant_values=0)
        print_matrix(balanced_matrix, msg=msg)

    def run_test(self):
        for index, matrix in enumerate(self.matrices + self.rectangular_matrices, start=1):
            print("\n----- TEST %d ------ " % (index))
            print_matrix(matrix, msg='original matrix')
            self.print_balanced_matrix(matrix, msg='\nbalanced matrix:')
            print("\nAnton's assignments")
            self.h.compute_assignments(matrix)
            pretty_print_assignments(self.h.matching, self.h.matrix)

            print("\nCorrects assignments")
            valid_assignments = self.compute_test_assignments(matrix) #  Will balance matrices using munkres module
            valid_profit = pretty_print_assignments(valid_assignments, matrix)
            print("\n")

            assert self.h.total_profit == valid_profit

def pretty_print_assignments(assignments, weights):
    """ Pretty prints assignments and it's weight
    :param assignments: assignments from node in X to node in Y. dict<int, int>
    :param weights: NxN profit matrix. Profit from X assigned to Y.
    :return: Total profit of assignments """
    total_profit = 0
    for key, value in assignments.items():
        weight = weights[key][value]
        total_profit += weight
        print('(%d, %d) -> %d' % (key, value, weight))

    print('total profit=%d' % total_profit)
    return total_profit

def main():

    workers = ["anton", "benjamin", "hugo", "viktor", "hankish", "dylan", "fredrik", "mattias", "björn"]
    jobs = ["clean", "wash", "paint", "attack", "mine", "scout", "build", "study", "eat"]
    matrix = [[1,2,3,4],[2,4,6,8],[3,6,9,12],[4,8,12,16]]
    matrices = []

    test = TestHungarian(max_problem_size = 20)
    test.run_test()


if __name__ == "__main__":
    main()
