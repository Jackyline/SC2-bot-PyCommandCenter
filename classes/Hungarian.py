class Hungarian(object):
    """
    Implementation of the Hungarian algorithm for the maximization assignment problem.
    Anton Hansson
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
        self._init_labels()
        self.matching = {}  # Let the matching be a dict from a vertex in X to a vertex in Y
        self.inverse_matching = {}
        self._find_and_augment()
        self.total_profit = sum(self.matrix[x][y] for x, y in self.matching.items())
        self.pretty_print()
        return self.matching

    def pretty_print(self):
        for n in range(len(self.matching)):
            self.matching[n] = self.jobs[self.matching[n]]
            self.matching[self.workers[n]] = self.matching.pop(n)

    def _init_labels(self):
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

    def _find_and_augment(self):
        """
        Core of the Hungarian algorithm. Find an augmenting path and augment the current matching.
         A solution is found when there is a perfect matching.
        :return: None
        """
        if len(self.matching) == self.n: # found perfect matching tror jag
            return

        # Find an unmatched vertex in X
        root = next(x for x in self.X if x not in self.matching)

        x, y, path = self._find_augmenting_path({root: None}, set([root]), set())
        self._augment_matching(x, y, path)
        self._find_and_augment()

    def _find_augmenting_path(self, path, S, T):
        """
        Find an augmenting path for the current matching. This may involve updating the feasible labelling
         in order to expand the equality graph and expose a vertex in Y that can be used to augment the matching.
        :param path: Traceable path to the root of the augmenting path.
                     Keys are vertices in X, values are the vertex in X preceding the key in the path to the root.
                     dict<int, int>
        :param S: set of vertices from X in the alternating tree. set<int>
        :param T: set of vertices from Y in the alternating tree. set<int>
        :return: a tuple (x, y, path) where (x, y) is the ending edge of the augmenting path and path is as above.
                 tuple(int, int, dict<int, int>)
        """
        for x in S:
            for y in self.Y:
                if not self._in_equality_graph(x, y):
                    continue

                if y in T:
                    continue  # Vertex already in the alternating tree

                if y not in self.inverse_matching:
                    return x, y, path  # Augmenting path has been found

                # Extend the alternating tree
                z = self.inverse_matching[y]
                S.add(z)
                T.add(y)
                path[z] = x
                return self._find_augmenting_path(path, S, T)

        # Neighbourhood of S is equal to T, update labelling to expose a vertex in Y
        self._update_labels(S, T)
        return self._find_augmenting_path(path, S, T)

    def _augment_matching(self, x, y, path):
        """
        Augments the current matching using the path ending with edge (x, y).
         (x, y) is not in the current matching. Neither is the root.
        :param x: last vertex in X in the augmenting path to the root. int
        :param y: very end of the augmenting path. int
        :param path: Traceable path to the root of the augmenting path.
                     Keys are vertices in X, values are the vertex in X preceding the key in the path to the root.
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
        self._augment_matching(path[x], matched_y, path)

    def _in_equality_graph(self, x, y):
        """
        Determine if edge (x, y) is in the equality graph.
        :param x: vertex from X. int
        :param y: vertex from Y. int
        :return: True if (x, y) is in the equality graph, False otherwise.
        """
        return self.matrix[x][y] == self.x_labels[x] + self.y_labels[y]

    def _update_labels(self, S, T):
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
        delta = None
        for x in S:
            for y in self.Y.difference(T):
                slack = self.x_labels[x] + self.y_labels[y] - self.matrix[x][y]
                if delta is None or slack < delta:
                    delta = slack

        for v in self.V:
            if v in S:
                self.x_labels[v] -= delta

            if v in T:
                self.y_labels[v] += delta

def main():

    workers = ["anton", "benjamin", "hugo", "viktor", "hankish", "dylan", "fredrik", "mattias", "björn"]
    jobs = ["clean", "wash", "paint", "attack", "mine", "scout", "build", "study", "eat"]
    matrix = [[7, 4, 3], [3, 1, 2], [3, 0, 0]]
    h = Hungarian(matrix, workers, jobs)
    print(h.compute())
    assert 9 == h.total_profit
    print(h.total_profit)

    matrix = [[2, 3, 3], [3, 2, 3], [3, 3, 2]]
    h = Hungarian(matrix, workers, jobs)
    print(h.compute())
    print(h.total_profit)

if __name__ == "__main__":
    main()