import hungarian
import numpy as np
from munkres import print_matrix
#import unit_manager

class TaskManager:

    #    def __init__(self, IDABot : IDABot):
            #self.idabot = IDABot

    def __init__(self):
        self.hungarian = hungarian.Hungarian()
        self.military_tasks = []
        self.workers_tasks = []
        self.building_tasks = []
        self.assignments = {}

    def add_task(task):
        pass

    def generate_matrix(self, utility_func, units, tasks):
        """
        Generates a profit matrix based on a utility function that is used to calculate the profit of assigning each task to each units. Profit matrix is made balanced but nr of dummy row/cols are saved to be removed later
        :param utility_func: Function that takes parameters :unit and :task and returns a profit (int) of assigning a task to a unit
        :param units: set of units represented by id. set<int>
        :param tasks: set of tasks represented as TODO
        :return: NxN profit matrix. Profit from unit X assigned to task Y.
        """

        n_units = len(units)
        n_tasks = len(tasks)

        matrix = np.zeros((n_units, n_tasks)) # This way balanced matrix is balanced right away
        for i in range(len(units)):
            for j in range(len(tasks)):
                matrix[i][j] = utility_func(units[i], tasks[j])

        return matrix

    def convert_matching_to_assignments(self, matching, units, tasks):
        """
        After computing assignments using hungarian the result will be a matching dict<int, int>, basically a mapping between integers representing workers and tasks. This function
        swap these integers back to the units and tasks in starcraft.
        :param matching: the matching returned by hungarian. dict<int, int>
        :param units: set of units represented by id. set<int>
        :param tasks: set of tasks represented as TODO
        :return: dict<TODO, TODO>
        """
        assignments = matching
        for n in range(len(assignments)):
            assignments[n] = tasks[assignments[n]]
            assignments[units[n]] = assignments.pop(n)
        return assignments

    def get_worker_tasks():
        return self.worker_tasks

    def svc_utility_func(self, unit, task):
        return 5

    def military_utility_func(self, groups, task):
        return 10

    def calc_assignments(self, utility_func, units, tasks):
        matrix = self.generate_matrix(utility_func, units, tasks)
        print_matrix(matrix, msg="\nGenerated svc matrix")
        matching = self.hungarian.compute_assignments(matrix)
        self.hungarian.pretty_print_assignments()
        assignments = self.convert_matching_to_assignments(matching, units, tasks)
        print(assignments)

    def update_assignments(self, new_assignments : dict):
        self.assignments.update(new_assignments)

def main():
    units = ["anton", "benjamin", "hugo", "viktor", "hankish", "dylan", "fredrik", "mattias", "björn"]
    tasks = ["clean", "wash", "paint", "attack", "mine", "scout", "build"]
    tasks2 = ["clean", "wash", "paint", "attack", "mine", "scout", "build", "bajsa", "dricka", "dansa", "skriva", "heja"]

    task_manager = TaskManager()
    task_manager.calc_assignments(task_manager.svc_utility_func, units, tasks)
    task_manager.calc_assignments(task_manager.military_utility_func, units, tasks2)


if __name__ == "__main__":
    main()
