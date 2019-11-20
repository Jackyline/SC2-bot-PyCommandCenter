class Hungarian(object):
    """
    Implementation of the Hungarian algorithm for the maximization assignment problem.
    """

    def __init__(self, matrix, workers, jobs):
        """
        Initialize instance of the assignment problem with the profit matrix
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
        self.slack = None

        def maximize(self):
            """
            Compute optimal matching for workers and jobs.

            step 1: Look for alternating path
                    blablablalb
            :return: maximum matching from X to Y. dict<int, int>
            """

            self.init_labels()
            self.matching = {}  # Let the matching be a dict with node in X as key with matching node in Y as value
            self.inverse_matching = {}


            while(len(self.matching) < n):

                # Find an unmatched node in X and set as root
                for x in self.X:
                    if x not in self.matching:
                        root = x
                        break

                path = {root: None}
                S = set([root])
                T = set()

                find_augmenting_path(path, S, T)

            return self.matching

        def is_in_equality_graph(self, x, y):
            """
            Determine if edge (x, y) is in the equality graph.
            :param x: node from X. int
            :param y: node from Y. int
            :return: True if (x, y) is in the equality graph, False otherwise.
            """
            return self.matrix[x][y] == self.x_labels[x] + self.y_labels[y]


        def find_augmenting_path(self, path, S, T):
            """
            Find an augmenting path for the current matching by building an alternating tree. If an augmenting path
            cannot be found the feasible labelling will be improved in order to expand the equality graph to find an augmenting path.
            :param path: Traceable path to the root of the augmenting path.
                         Keys are nodes in X, values are the node in X preceding the key in the path to the root.
                         dict<int, int>
            :param S: set of nodes from X in the alternating tree. set<int>
            :param T: set of nodes from Y in the alternating tree. set<int>
            :return: a tuple (x, y, path) where (x, y) is the ending edge of the augmenting path and path is as above.
                     tuple(int, int, dict<int, int>)
            """

            while (True): # Expand the alternating tree until augmented path is found
                for s in S:
                    for y in self.Y:
                        if not self.is_in_equality_graph(s, y):
                            continue  # Results in iteration for each neighbouring node to each node in S along edges in
                            # equality graph.

                        if y in T:
                            continue  # Node already in the alternating tree

                        # We have found an y so that y is a neighbour of s but not in T
                        # Now check if y is matched or unmatched

                        if y not in self.inverse_matching:  # y is unmatched / exposed / free -> Augmenting path has been found
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

            self.matching[x] = y
            self.inverse_matching[y] = x

            while (path[x] is not None):
                matched_y = self.matching[x]
                self.matching[x] = y
                self.inverse_matching[y] = x
                x = path[x]
                y = matched_y
            return

        def improve_labels(self, S, T):
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