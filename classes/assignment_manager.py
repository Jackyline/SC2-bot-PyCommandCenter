import math

import numpy as np
from classes.building_manager import BuildingManager
from classes.hungarian import Hungarian
from classes.task_type import TaskType
from classes.unit_manager import UnitManager
from library import *
from classes.task import Task

from munkres import print_matrix


class AssignmentManager:

    def __init__(self, ida_bot):
        self.ida_bot = ida_bot
        self.building_manager = ida_bot.building_manager
        self.unit_manager = ida_bot.unit_manager
        self.worker_assignments = WorkerAssignments(self.unit_manager)
        self.military_assignments = MilitaryAssignments(self.ida_bot)
        self.building_assignments = BuildingAssignments(self.building_manager)
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
        assignments = {}
        for worker_nr, task_nr in matching.items():
            assignments[tasks[task_nr]] = units[worker_nr]
        return assignments

    def calc_assignments(self, task_type):
        all_tasks = task_type.get_tasks()
        available_units = task_type.get_available_units()
        matrix = self.generate_matrix(task_type.utility_func, available_units, all_tasks)
        #print_matrix(matrix, msg="Matrix generated for: " + task_type.toString())
        #print("### CALCULATING ASSIGNMENTS ### \n Nr tasks: ", len(all_tasks), "\n Nr units: ", len(available_units))
        matching = self.hungarian.compute_assignments(matrix)
        #self.hungarian.pretty_print_assignments()
        assignments = self.convert_matching_to_assignments(matching, available_units, all_tasks)
        task_type.tasks.clear()
        return assignments

    def update_assignments(self, assignment_type):
        """
        Generic function that calls calc_assignments with tasks and units relevant to the specific assignment_type
        as input. Calculated assignments are then saved to the list assignments in the specific assignment_type class
        :param assignment_type: WorkerAssignments, Militaryassignments or BuildingAssignments.
        """

        if assignment_type.tasks and len(assignment_type.get_available_units()) > 0:  # Make sure there are new tasks and units that can do them
            assignments = self.calc_assignments(assignment_type)
            assignment_type.update(assignments)
        else:
            assignment_type.tasks.clear()

    def on_step(self):
        if self.ida_bot.current_frame % 10 == 0:
            return
        # Add recommended nr of gas and mining tasks
        self.generate_gas_tasks()
        self.generate_mining_tasks()

        # Update worker assignments
        self.update_assignments(self.worker_assignments)

        # Update military assignments
        self.update_assignments(self.military_assignments)

        # Update building assignments
        self.update_assignments(self.building_assignments)


    def generate_mining_tasks(self):
        nr_mining_jobs = len(self.unit_manager.worker_units)
        if self.worker_assignments.get_available_units():  # Only generate if there are available workers
            for base in sorted(self.ida_bot.base_location_manager.get_occupied_base_locations(PLAYER_SELF),key= lambda x: x.is_start_location, reverse=True):
                for i in range(2*len(self.ida_bot.get_mineral_fields(base))):
                    self.worker_assignments.add_task(Task(task_type=TaskType.MINING,
                                                          pos=Point2D(base.depot_position.x, base.depot_position.y),
                                                          base_location=base))
                    nr_mining_jobs -= 1
                    if nr_mining_jobs == 0:
                        return

    def generate_gas_tasks(self):
        if self.worker_assignments.get_available_units(): # Only generate if there are available workers
            for refinary in self.building_manager.get_buildings_of_type(UnitType(UNIT_TYPEID.TERRAN_REFINERY, self.ida_bot)):
                for i in range(2):
                    self.worker_assignments.add_task(Task(task_type=TaskType.GAS, pos=refinary.get_unit().position))

    def add_task(self, task):
        """
        Adds a task to the UNIT_assignment, where UNIT can be worker, military group or building
        """

        # Tasks done by workers
        if task.task_type is TaskType.MINING or task.task_type is TaskType.GAS or task.task_type is TaskType.BUILD or task.task_type is TaskType.SCOUT:
            self.worker_assignments.add_task(task)

        # Tasks done by military units
        elif task.task_type is TaskType.ATTACK or task.task_type is TaskType.DEFEND:
            self.military_assignments.add_task(task)

        # Tasks done by buildings
        elif task.task_type is TaskType.TRAIN or task.task_type is TaskType.ADD_ON:
            self.building_assignments.add_task(task)


class WorkerAssignments:

    def __init__(self, unit_manager: UnitManager):
        self.unit_manager = unit_manager
        self.assignments = {}  # dict<task, worker_unit>
        self.tasks = []
        self.currently_mining = 0

    def utility_func(self, worker, task):
        Point2D.distance = lambda self, other: math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
        distance = int(task.pos.distance(worker.get_unit().position))
        idle = worker.is_idle()

        profit = 0
        if not worker.task is None and worker.task == task: # valuable to do the same task as before
            profit += 1000
        if task.task_type == TaskType.SCOUT:
            profit += 10000
        elif task.task_type == TaskType.BUILD:
            profit += 10000
        elif task.task_type is TaskType.GAS:
            profit += 1000
        if idle:
            profit += 100

        profit -= distance



        if profit < 0:
            profit = 0

        return profit

    def get_available_units(self):
        available_units = []
        for worker in self.unit_manager.worker_units:
            task = worker.get_task()
            if task is None:
                available_units.append(worker) # worker is available
            elif task.task_type is TaskType.MINING or task.task_type is TaskType.GAS: # also add those who are mining or collecting gas
                available_units.append(worker)
        return available_units

    def toString(self):
        return "worker assignments"

    def update(self, new_assignments: dict):
        workers_that_previusly_were_mining_or_gassing_but_no_longor_will =  []
        for task in self.assignments:
            if task.task_type is not TaskType.MINING and task.task_type is not TaskType.GAS:
                new_assignments[task] = self.assignments[task]
            else:
                workers_that_previusly_were_mining_or_gassing_but_no_longor_will.append(self.assignments[task])

        for worker in workers_that_previusly_were_mining_or_gassing_but_no_longor_will:
            if not worker in new_assignments.values():
                worker.set_task(None)
                worker.set_idle()

        self.assignments = new_assignments
        for worker_unit in self.get_available_units():
            for task, assigned_unit in self.assignments.items():
                if worker_unit.get_id() == assigned_unit.get_id():
                    prev_task = worker_unit.get_task()
                    if prev_task is None or not prev_task == task:
                        worker_unit.set_task(task)
                        self.unit_manager.command_unit(worker_unit, task)

    def add_task(self, task):
        self.tasks.append(task)

    def add_already_assigned_tasks(self):
        """
        Add all mining and gas assignments as new tasks in order to reevaluate who should do what. Sometimes it might be
        profitable to have a worker that is already assigned to mining to construct something nearby than having an
        idle worker running a long distance.
        """
        self.remove_finished_tasks()
        for task in self.assignments:
            if task.task_type is TaskType.MINING or task.task_type is TaskType.GAS:
                self.tasks.append(task)

    def remove_finished_tasks(self):
        to_remove = {}
        for task, worker in self.assignments.items():

            if worker.get_unit().is_idle:
                if not worker.get_task().task_type is TaskType.SCOUT:
                    to_remove[task] = worker

            elif not worker.is_alive():
                to_remove[task] = worker
            # TODO: fix worker task when refinery is done
            #elif worker.task.construct_building.is_refinery and worker.task.construct_building.is_complete and worker.task.task_type is TaskType.BUILD:

        for task, worker in to_remove.items():
            worker.set_task(None)
            self.assignments.pop(task)

    def get_tasks(self):
        self.remove_finished_tasks()
        #self.add_already_assigned_tasks()
        return self.tasks


class MilitaryAssignments:

    def __init__(self, IDABot: IDABot):
        self.IDABot = IDABot
        self.unit_manager = IDABot.unit_manager
        self.assignments = {}  # dict<task, worker_unit>
        self.tasks = []

    def utility_func(self, group, task):
        #TODO: här måste vi kolla att det är mkt reward om den kan bygga, lite annars
        return 5

    def get_available_units(self):
        #if not len(self.tasks) == len() # If strategy has changed, or number of base locations has changed and we are in defensive strategy, create new groups
        #self.unit_manager.create_coalition()
        #else:
            #self.unit_manager.add_units_to_coalition()
        return self.unit_manager.groups

    def toString(self):
        return "military assignments"

    def update(self, new_assignments: dict):
        self.assignments = new_assignments
        for task, group in new_assignments.items():
            self.unit_manager.command_group(task, group)

    def add_task(self, task):
        self.tasks.append(task)

    def get_tasks(self):
        return self.tasks


class BuildingAssignments:

    def __init__(self, building_manager: BuildingManager):
        self.building_manager = building_manager
        self.assignments = {}  # dict<task, worker_unit>
        self.tasks = []

    def utility_func(self, building, task):

        profit = 0
        if task.task_type is TaskType.TRAIN:
            if building.get_unit() in self.building_manager.get_my_producers(task.produce_unit):
                profit += 1000
        elif task.task_type is TaskType.ADD_ON:
            if building.get_unit() in self.building_manager.get_my_producers(task.construct_building):
                profit += 1000
        if building.is_training:
            profit -= 100

        if profit < 0:
            profit = 0
        return profit

    def get_available_units(self):
        return [building for building in self.building_manager.buildings if not building.get_unit().is_training]

    def toString(self):
        return "building assignments"

    def add_task(self, task):
        self.tasks.append(task)

    def update(self, new_assignments: dict):
        self.assignments = new_assignments
        for task, building in self.assignments.items():

            self.building_manager.command_building(building, task)

    def get_tasks(self):
        return self.tasks
