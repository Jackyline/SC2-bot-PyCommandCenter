from munkres import Munkres, print_matrix
from random import randint
import numpy as np
import sys
from hungarian import *

class TestHungarian():
    """
    Imported Munkres module that provides an implementation of the Munkres algorithm
    that can be used to test my implementation. Generates 10 balanced matrices and 10
    unbalanced matrices of maximum size 'max_problem_size that tests are run on

    How to install munkres test module:
    try: 'pip install munkres3'
    If pip install does not work the module named 'munkres3-1.0.5.5' exists in the same folder as this file.
    When inside folder run: 'python3 setup.py install'
    """

    def __init__(self, max_problem_size):
        self.h = Hungarian()
        self.m = Munkres()
        self.matrices = []
        self.rectangular_matrices = []

        # Generate random matrices used for testing
        for i in range(1, 11):
            rand_int1 = randint(1, max_problem_size)
            rand_int2 = randint(1, max_problem_size)
            self.matrices.append(np.random.randint(max_problem_size, size=(rand_int1, rand_int1)))  # balanced matrices
            self.rectangular_matrices.append(np.random.randint(max_problem_size, size=(rand_int1, rand_int2)))  # unbalanced matrices

    def pretty_print_assignments(self, assignments, matrix):
        """ Pretty prints assignments and it's profit
        :param assignments: assignments from node in X to node in Y. dict<int, int>
        :param matrix: NxN profit matrix. Profit from X assigned to Y.
        :return: Total profit of assignments """
        total_profit = 0
        for key, value in assignments.items():
            profit = matrix[key][value]
            total_profit += profit
            print('(%d, %d) -> %d' % (key, value, profit))

        print('total profit=%d' % total_profit)
        return total_profit

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
        """ Runs anton's implementation of the hungarian algorithm and compares the results to the results of the imported munkres module. The munkres module will balance the problem automatically. Anton's implementation has it's own balancing function. Assignments can differ between the two algorithms, but the total the number of assignments and total profit should always be equal.
        """
        test_matrices = self.matrices + self.rectangular_matrices

        for index, matrix in enumerate(test_matrices, start=1):
            print("\n\n----- TEST %d ------ " % (index))
            print_matrix(matrix, msg='Original matrix:')
            self.print_balanced_matrix(matrix, msg='\nBalanced matrix:')

            print("\nAnton's assignments:")
            self.h.compute_assignments(matrix)
            self.h.pretty_print_assignments()

            print("Correct assignments:")
            valid_assignments = self.compute_test_assignments(matrix) #  will balance matrices using munkres module
            valid_profit = self.pretty_print_assignments(valid_assignments, matrix)

            assert self.h.total_profit == valid_profit  # make sure total profit is correct
            assert len(self.h.matching) == len(valid_assignments) # make sure that dummy row/columns have been removed

def main():
    test = TestHungarian(max_problem_size = 50)
    test.run_test()

if __name__ == "__main__":
    main()
