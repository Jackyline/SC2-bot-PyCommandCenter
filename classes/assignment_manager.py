import numpy as np
from classes.building_manager import BuildingManager
from classes.hungarian import Hungarian
from classes.task_type import TaskType
from classes.unit_manager import UnitManager
from munkres import print_matrix


class WorkerAssignments():
    """
    TODO
    """

    def __init__(self, unit_manager: UnitManager):
        self.unit_manager = unit_manager
        self.assignments = {}  # dict<task, worker_unit>
        self.tasks = []

    def utility_func(self, worker, task):
        return 3

    def get_all_units(self):
        return self.unit_manager.worker_units

    def toString(self):
        return "worker assignments"

    def update(self, new_assignments: dict):
        self.assignments.update(new_assignments)
        for worker_unit in self.get_all_units():
            for task, assigned_unit in self.assignments.items():
                if worker_unit == assigned_unit:
                    prev_task = worker_unit.get_task()
                    if prev_task != task:
                        worker_unit.set_task(task)
                        # TODO utför jobbet också!
                        self.unit_manager.command_unit(worker_unit, task)

    def add_task(self, task):
        self.tasks.append(task)

    def add_already_assigned_tasks(self):
        for key in self.assignments:
            self.tasks.append(key)


class MilitaryAssignments():
    """
    TODO
    """

    def __init__(self, unit_manager: UnitManager):
        self.unit_manager = unit_manager
        self.assignments = {}  # dict<task, worker_unit>
        self.tasks = []

    def utility_func(self, group, task):
        return 5

    def get_all_units(self):
        return self.unit_manager.military_units

    def toString(self):
        return "military assignments"

    def update(self, new_assignments: dict):
        self.assignments.update(new_assignments)
        for military_unit in self.get_all_units():
            for task, assigned_unit in self.assignments:
                if military_unit == assigned_unit:
                    military_unit.set_task(task)

    def add_task(self, task):
        self.tasks.append(task)

    def add_already_assigned_tasks(self):
        for key in self.assignments:
            self.tasks.append(key)


class BuildingAssignments:
    """
    TODO
    """

    def __init__(self, building_manager: BuildingManager):
        self.building_manager = building_manager
        self.assignments = {}  # dict<task, worker_unit>
        self.tasks = []

    def utility_func(self, building, task):
        return 10

    def get_all_units(self):
        return self.building_manager.buildings  # TODO kan antalet units förändras under en körning?

    def toString(self):
        return "building assignments"

    def add_task(self, task):
        self.tasks.append(task)

    def update(self, new_assignments: dict):
        self.assignments = new_assignments
        for building_unit in self.get_all_units():
            for task, assigned_building in self.assignments:
                if building_unit == assigned_building:
                    building_unit.set_task(task)

    def add_already_assigned_tasks(self):
        for key in self.assignments:
            self.tasks.append(key)


class AssignmentManager:

    def __init__(self, unit_manager: UnitManager, building_manager: BuildingManager):
        self.worker_assignments = WorkerAssignments(unit_manager)
        self.military_assignments = MilitaryAssignments(unit_manager)
        self.building_assignments = BuildingAssignments(building_manager)
        self.hungarian = Hungarian()

    def generate_matrix(self, utility_func, units, tasks):
        """
        Generates a profit matrix based on a utility function that is used to calculate the profit of assigning each task to each units. Profit matrix is made balanced but nr of dummy row/cols are saved to be removed later
        :param utility_func: Function that takes parameters :unit and :task and returns a profit (int) of assigning a task to a unit
        :param units: set of worker_unit or military_unit objects
        :param tasks: set of task objects
        :return: NxN profit matrix. Profit from unit X assigned to task Y.
        """

        n_units = len(units)
        n_tasks = len(tasks)

        matrix = np.zeros((n_units, n_tasks))  # This way balanced matrix is balanced right away
        for i in range(len(units)):
            for j in range(len(tasks)):
                matrix[i][j] = utility_func(units[i], tasks[j])

        return matrix

    def convert_matching_to_assignments(self, matching, units, tasks):
        """
        After computing assignments using hungarian the result will be a matching dict<int, int>, basically a mapping between integers representing workers and tasks. This function
        swap these integers back to the units and tasks in starcraft. previously the X coordinates of the matrix have been units and the Y coord tasks. These are now swapped to allow
        tasks being easily found in the dict.
        :param matching: the matching returned by hungarian. dict<int, int>
        :param units: set of worker_unit or military_unit
        :param tasks: set of task objects
        :return: dict<task, worker_unit / military_unit / building_unit>
        """
        assignments = matching
        for n in range(len(assignments)):
            assignments[n] = units[assignments[n]]
            assignments[tasks[n]] = assignments.pop(n)
        return assignments

    def calc_assignments(self, task_type):
        """
        TODO
        :param task_type:
        :return:
        """
        task_type.add_already_assigned_tasks()
        matrix = self.generate_matrix(task_type.utility_func, task_type.get_all_units(), task_type.tasks)
        print_matrix(matrix, msg="Matrix generated for: " + task_type.toString())
        matching = self.hungarian.compute_assignments(matrix)
        self.hungarian.pretty_print_assignments()
        assignments = self.convert_matching_to_assignments(matching, task_type.get_all_units(), task_type.tasks)
        task_type.tasks.clear()
        return assignments

    def update_assignments(self, assignment_type):
        """
        Generic function that calls calc_assignments with tasks and units relevant to the specific assignment_type
        as input. Calculated assignments are then saved to the list assignments in the specific assignment_type class
        :param assignment_type: WorkerAssignments, Militaryassignments or BuildingAssignments.
        """

        if assignment_type.tasks:  # Make sure there are new tasks and units that can do them
            if len(assignment_type.get_all_units()) > 0:  # TODO byggnader? ska kanske kolla can produce?
                assignments = self.calc_assignments(assignment_type)
                assignment_type.update(assignments)

    def on_step(self):
        """
        TODO
        """
        # Update worker assignments
        self.update_assignments(self.worker_assignments)

        # Update military assignments
        self.update_assignments(self.military_assignments)

        # Update building assignments
        self.update_assignments(self.building_assignments)

    def add_task(self, task):
        """
        Adds a task to the UNIT_assignment, where UNIT can be worker, military or building, based on the task_type of the task
        """
        # Tasks done by workers
        if task.task_type is TaskType.MINING or task.task_type is TaskType.GAS or task.task_type is TaskType.BUILD:
            self.worker_assignments.add_task(task)

        # Tasks done by military units
        elif task.task_type is TaskType.ATTACK:
            self.military_assignments.add_task(task)

        # Tasks done by buildings
        elif task.task_type is TaskType.TRAIN:
            self.building_assignments.add_task(task)
