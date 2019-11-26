import hungarian
import numpy as np
from munkres import print_matrix
#import unit_manager

class WorkerAssignments():
    def __init__(self):
        self.assignments = {}
        self.tasks = []

    def utility_func(self, worker, task):
        return 3

    def get_all_units(self):
        pass

    def toString(self):
        return "worker assignments"

    def update(self, new_assignments : dict):
        self.assignments.update(new_assignments)

    def add_task(self, task):
        self.tasks.append(task)

class MilitaryAssignments():
    def __init__(self):
        self.assignments = {}
        self.tasks = []

    def utility_func(self, group, task):
        return 5

    def get_all_units(self):
        pass

    def toString(self):
        return "military assignments"

    def update(self, new_assignments : dict):
        self.assignments.update(new_assignments)

    def add_task(self, task):
        self.tasks.append(task)

class BuildingAssignments():
    def __init__(self):
        self.assignments = {}
        self.tasks = []

    def utility_func(self, building, task):
        return 10

    def get_all_units(self):
        pass

    def toString(self):
        return "building assignments"

    def add_task(self, task):
            self.tasks.append(task)

    def update(self, new_assignments : dict):
        """
        for key, value in new_assignments:

        for key, value in self.assignments:
            if value in

        """
        self.assignments.update(new_assignments)

class TaskManager():

    #    def __init__(self, IDABot : IDABot):
            #self.idabot = IDABot

    def __init__(self):
        self.worker_assignments = WorkerAssignments()
        self.military_assignments = MilitaryAssignments()
        self.building_assignments = BuildingAssignments()
        self.hungarian = hungarian.Hungarian()

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

    def calc_assignments(self, task_type):
        #units = task_type.get_all_units()
        units = ["anton", "benjamin", "hugo", "viktor", "hankish", "dylan", "fredrik", "mattias", "björn"] # instead of get all units
        matrix = self.generate_matrix(task_type.utility_func, units, task_type.tasks)
        print_matrix(matrix, msg="Matrix generated for: " + task_type.toString())
        matching = self.hungarian.compute_assignments(matrix)
        self.hungarian.pretty_print_assignments()
        assignments = self.convert_matching_to_assignments(matching, units, task_type.tasks)
        task_type.tasks.clear()
        return assignments

    def update_assignments(self, task_type):
        if task_type.tasks:
            print("\n" + task_type.toString() + " before update", task_type.assignments, "\n")
            assignments = self.calc_assignments(task_type)
            task_type.update(assignments)
            print(task_type.toString() + " after update", task_type.assignments)

    def on_step(self):
        # Update worker assignments
        self.update_assignments(self.worker_assignments)

        # Update military assignments
        self.update_assignments(self.military_assignments)

        # Update building assignments
        self.update_assignments(self.building_assignments)

        pass

    def add_worker_task(self, task):
        self.worker_assignments.add_task(task)

    def add_military_task(self, task):
        self.military_assignments.add_task(task)

    def add_building_task(self, task):
        self.building_assignments.add_task(task)

def main():
    tasks = ["clean", "wash", "paint", "attack", "mine", "scout", "build"]
    tasks2 = ["clean", "wash", "paint", "attack", "mine", "scout", "build", "bajsa", "dricka", "dansa", "skriva", "heja"]

    task_manager = TaskManager()
    for task in tasks:
        task_manager.add_worker_task(task)

    for task in tasks2:
        task_manager.add_military_task(task)

    task_manager.on_step()

if __name__ == "__main__":
    main()
