class Hungarian(object):
    """
    Implementation of the Hungarian algorithm for the maximization assignment problem.
    """
    def __init__(self, matrix, workers, jobs):
        """
        Initialize instance of the assignment problem with the profit matrix `matrix`
        :param matrix: NxN profit matrix. Profit from X assigned to Y.
        """

        # TODO: create profit matrix with help of utility function, set n to nr of jobs/workers
        n = len(matrix)
        for r in matrix:
            if len(r) != n:
                # TODO: create dummy worker or job
                raise ValueError('Hungarian algorithm accepts an NxN matrix.')

        self.workers = workers
        self.jobs = jobs
        self.matrix = matrix
        self.n = n
        self.V = self.X = self.Y = set(range(n))  # For convenience and clarity
        self.x_labels = None
        self.y_labels = None
        self.matching = None
        self.inverse_matching = None
        self.total_profit = None
        self.minSlack = None

    def maximize(self):
        """
        :return: maximum matching from X to Y. dict<int, int>
        """
        return self.compute()


    def compute(self):
        """
        Compute optimal matching for workers and jobs
        :return: optimal matching from X to Y. dict<int, int>
        """
        self.init_labels()
        self.matching = {}  # Let the matching be a dict with node in X as key with matching node in Y as value
        self.inverse_matching = {}
        self.find_augmented_path_and_augment()
        self.total_profit = sum(self.matrix[x][y] for x, y in self.matching.items())
        # self.pretty_print()
        return self.matching

    def pretty_print(self):
        for n in range(len(self.matching)):
            self.matching[n] = self.jobs[self.matching[n]]
            self.matching[self.workers[n]] = self.matching.pop(n)

    def init_labels(self):
        """
        Initialize the labelling for each node x to the max weight of all it's edges.
        Initialize the labelling for each node y to 0.
        :return: None
        """
        self.x_labels = [0 for _ in self.X]
        self.y_labels = [0 for _ in self.Y]
        for x in self.X:
            for y in self.Y:
                self.x_labels[x] = max(self.x_labels[x], self.matrix[x][y])

    def slack(self, x, y):
        return self.x_labels[x] + self.y_labels[y] - self.matrix[x][y]

    def find_augmented_path_and_augment(self):
        """
        Core of the Hungarian algorithm. Find an augmenting path and augment the current matching.
         A solution is found when there is a perfect matching.
        :return: None
        """
        if len(self.matching) == self.n:
            # Has found a perfect matching
            return

        # Find an unmatched node in X and set as root
        for x in self.X:
            if x not in self.matching:
                root = x
                break

        self.minSlack = [[self.slack(root, y), root] for y in self.Y]
        x, y, path = self.find_augmenting_path({root: None}, set([root]), set())
        self.augment_matching(x, y, path)
        self.find_augmented_path_and_augment()

    def find_augmenting_path(self, path, S, T):
        """
        Find an augmenting path for the current matching. If an augmenting path cannot be found the feasible labelling
        will be improved in order to expand the equality graph to find an augmenting path.
        :param path: Traceable path to the root of the augmenting path.
                     Keys are nodes in X, values are the node in X preceding the key in the path to the root.
                     dict<int, int>
        :param S: set of nodes from X in the alternating tree. set<int>
        :param T: set of nodes from Y in the alternating tree. set<int>
        :return: a tuple (x, y, path) where (x, y) is the ending edge of the augmenting path and path is as above.
                 tuple(int, int, dict<int, int>)
        """


        """ Part from skit.py """
        while True:

            # select edge (x,y) with x in S, y not in T and min slack
            ((val, x), y) = min([(self.minSlack[y], y) for y in self.Y if y not in T])
            assert x in S
            if val > 0:
                self.improve_labels(TODO)
            assert self.slack(x, y) == 0  # by now the found y should be part of equality graph, which means slack = 0

            if y in self.inverse_matching:  # y is matched -> Extend the alternating tree
                z = self.inverse_matching[y]
                S.add(z)
                T.add(y)
                path[z] = s

            else:  # y is unmatched -> Augmenting path has been found
                return s, y, path
            """ end part from skit.py """




            # Expand the alternating tree until augmented path is found
            for s in S:
                for y in self.Y:
                    if not self.is_in_equality_graph(s, y):
                        continue  # Results in iteration for each neighbouring node to each node in S along edges in
                        # equality graph.

                    if y in T:
                        continue  # Node already in the alternating tree

                    # We have found an y so that y is a neighbour of s but not in T
                    # Now check if y is matched or unmatched

                    if y not in self.inverse_matching:  # y is unmatched -> Augmenting path has been found
                        return s, y, path

                    # y is matched -> Extend the alternating tree
                    z = self.inverse_matching[y]
                    S.add(z)
                    T.add(y)
                    path[z] = s
                    return self.find_augmenting_path(path, S, T)

            # Neighbourhood of S is equal to T, so we cannot increase the alternating path.
            # Instead improve labelling
            self.improve_labels(S, T)


    def augment_matching(self, x, y, path):
        """
        Augments the current matching using the path ending with edge (x, y).
         (x, y) is not in the current matching. Neither is the root.
        :param x: last node in X in the augmenting path to the root. int
        :param y: very end of the augmenting path. int
        :param path: Traceable path to the root of the augmenting path.
                     Keys are nodes in X, values are the nodes in X preceding the key in the path to the root.
                     dict<int, int>
        :return: None
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
        Determine if edge (x, y) is in the equality graph.
        :param x: node from X. int
        :param y: node from Y. int
        :return: True if (x, y) is in the equality graph, False otherwise.
        """
        return self.matrix[x][y] == self.x_labels[x] + self.y_labels[y]

    def improve_labels(self, val, S):
        """
        Improve the current labelling such that:
            - the current matching remains in the new equality graph
            - the current alternating tree (path) remains in the new equality graph
            - there is a free vertex from Y and not in T in the new equality graph
        An assumption is made that the neighbourhood of S in the equality graph is equal to T.

        :param S: set of vertices from X in the alternating tree. set<int>
        :param T: set of vertices from Y in the alternating tree. set<int>
        :return: None
        """

        """ TODO DU HÅLLER PÅ HÄÄR OCH SKA LIKNA DENNA FUNKTION TILL DEN I skit.py, ANNARS GJORDE DU PRECIS KLAR 
        find_augmenting_path """
        for v in self.V:
            if v in S:
                self.x_labels[v] -= delta

            if v in T:
                self.y_labels[v] += delta

def main():

    workers = ["anton", "benjamin", "hugo", "viktor", "hankish", "dylan", "fredrik", "mattias", "björn"]
    jobs = ["clean", "wash", "paint", "attack", "mine", "scout", "build", "study", "eat"]
    matrix = [[1,2,3,4],[2,4,6,8],[3,6,9,12],[4,8,12,16]]
    h = Hungarian(matrix, workers, jobs)
    print(h.compute())
    assert 30 == h.total_profit
    print(h.total_profit)

    matrix = [[25, 3, 3], [3, 2, 3], [3, 3, 2]]
    h = Hungarian(matrix, workers, jobs)
    print(h.compute())
    print(h.total_profit)

if __name__ == "__main__":
    main()