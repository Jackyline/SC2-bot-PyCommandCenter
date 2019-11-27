from classes.hungarian import Hungarian
import numpy as np
from munkres import print_matrix
from library import *
from classes.unit_manager import UnitManager
from classes.building_manager import BuildingManager
from classes.Task import Task


bajsTask(Task.NOTHING)

class bajsTask(Enum):
    def __init__(self, type):
        self.type = TaskType.NOTHING
        self.pos = Unit.tile_position



class WorkerAssignments():
    def __init__(self, unit_manager: UnitManager):
        self.unit_manager = unit_manager
        self.assignments = {} # dict<task, worker_unit>
        self.tasks = []


    def utility_func(self, worker, task):
        return 3

    def get_all_units(self):
        return self.unit_manager.get_all_workers()

    def toString(self):
        return "worker assignments"

    def update(self, new_assignments : dict):
        self.assignments.update(new_assignments)

    def add_task(self, task):
        self.tasks.append(task)

class MilitaryAssignments():
    def __init__(self, unit_manager: UnitManager):
        self.unit_manager = unit_manager
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
    def __init__(self, building_manager : BuildingManager):
        self.building_manager = building_manager
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

    def __init__(self, unit_manager: UnitManager, building_manager: BuildingManager):
        self.unit_manager = UnitManager # kanske kan ta bort
        self.building_manager = BuildingManager # kanske kan ta bort
        self.worker_assignments = WorkerAssignments(unit_manager)
        self.military_assignments = MilitaryAssignments(unit_manager)
        self.building_assignments = BuildingAssignments(building_manager)
        self.hungarian = Hungarian()

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

    def calc_assignments(self, task_type):

        units = ["anton", "benjamin", "hugo", "viktor", "hankish", "dylan", "fredrik", "mattias", "björn"] # instead of get all units
        matrix = self.generate_matrix(task_type.utility_func, task_type.get_all_units(), task_type.tasks)
        print_matrix(matrix, msg="Matrix generated for: " + task_type.toString())
        matching = self.hungarian.compute_assignments(matrix)
        self.hungarian.pretty_print_assignments()
        assignments = self.convert_matching_to_assignments(matching, task_type.get_all_units(), task_type.tasks)
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
        Task()
        self.update_assignments(self.worker_assignments)

        # Update military assignments
        #self.update_assignments(self.military_assignments)

        # Update building assignments
        #self.update_assignments(self.building_assignments)

    def add_worker_task(self, task):
        self.worker_assignments.add_task(task)

    def add_military_task(self, task):
        self.military_assignments.add_task(task)

    def add_building_task(self, task):
        self.building_assignments.add_task(task)
