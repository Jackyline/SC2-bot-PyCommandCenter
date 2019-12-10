from classes.military_unit import MilitaryUnit
from classes.worker_unit import WorkerUnit
from classes.q_table import QTable
from classes.coalitionstructure_generation import CoalitionstructureGenerator
from classes.task_type import TaskType
from classes.scout_unit import ScoutUnit
from classes.resource_manager import ResourceManager
from library import *
import math
import random

class UnitManager:

    def __init__(self, idabot):
        self.idabot = idabot
        # All military types
        self.MILITARY_TYPES = [UnitType(UNIT_TYPEID.PROTOSS_STALKER, idabot),
                               UnitType(UNIT_TYPEID.TERRAN_MARINE, idabot),
                               UnitType(UNIT_TYPEID.TERRAN_MARAUDER, idabot),
                               UnitType(UNIT_TYPEID.TERRAN_REAPER, idabot),
                               UnitType(UNIT_TYPEID.TERRAN_GHOST, idabot),
                               UnitType(UNIT_TYPEID.TERRAN_HELLION, idabot),
                               UnitType(UNIT_TYPEID.TERRAN_SIEGETANK, idabot),
                               UnitType(UNIT_TYPEID.TERRAN_CYCLONE, idabot),
                               UnitType(UNIT_TYPEID.TERRAN_WIDOWMINE, idabot),
                               UnitType(UNIT_TYPEID.TERRAN_THOR, idabot),
                               UnitType(UNIT_TYPEID.TERRAN_VIKINGASSAULT, idabot),
                               UnitType(UNIT_TYPEID.TERRAN_VIKINGFIGHTER, idabot),
                               UnitType(UNIT_TYPEID.TERRAN_MEDIVAC, idabot),
                               UnitType(UNIT_TYPEID.TERRAN_LIBERATOR, idabot),
                               UnitType(UNIT_TYPEID.TERRAN_RAVEN, idabot),
                               UnitType(UNIT_TYPEID.TERRAN_BANSHEE, idabot),
                               UnitType(UNIT_TYPEID.TERRAN_BATTLECRUISER, idabot),
                               UnitType(UNIT_TYPEID.TERRAN_AUTOTURRET, idabot)]
        # Worker types
        self.WORKER_TYPES = [UnitType(UNIT_TYPEID.TERRAN_SCV, idabot)]

        # List of our abstracted worker units
        self.worker_units = []

        # List of out abstracted scout units
        self.scout_units = []

        # List of our abstracted military units
        self.military_units = []

        # list of visible military units
        self.visible_enemies = []

        self.marauder_q_table = QTable(self.idabot, "marauder")
        self.marine_q_table = QTable(self.idabot, "marine")
        self.helion_q_table = QTable(self.idabot, "helion")
        self.cyclone_q_table = QTable(self.idabot, "cyclone")

        # Keeps track of current coalition structure, structured as [[id1, id2, ...], [id1, id2...], ...]
        self.csg = CoalitionstructureGenerator()
        self.groups = []

    def get_info(self):
        '''
        :return: Info about all units and their type/workstation
        '''
        info = {}

        # {job_type : amount doing that job type}
        info["workerUnits"] = {task_type : len([worker for worker in self.worker_units if not worker.get_task() is None and worker.get_task().task_type == task_type]) for task_type in TaskType}
        """
        info["workerUnits"] = {worker.get_task().task_type if worker.get_task() else None:
                                   len([task for task in self.idabot.assignment_manager.worker_assignments.assignments if task.task_type == worker.get_task().task_type])
                               for worker in self.worker_units
                               if "workerUnits" not in info
                               or worker.get_job() not in info["workerUnits"]
                               }
        """
        # {Military_type : amount of units of given type}
        info["militaryUnits"] = {
            military_type.get_unit_type(): len(self.get_units_of_type(military_type.get_unit_type()))
            for military_type in self.military_units
            if "militaryUnits" not in info
               or military_type.get_unit_type() not in info["militaryUnits"]
        }

        return info

    def get_units_working_on_type(self, workstation):
        '''
        :param workstation: Unit type building or workstation
        :return: All units that have their current job set to :param workstation
        '''
        return [worker for worker in self.worker_units
                if (worker.get_job() and workstation and worker.get_job().unit_type == workstation.unit_type)
                or (worker.get_job() == workstation)]

    def get_units_of_type(self, unit_type):
        """
        Gets all the unit with the unit type.
        :param unit_type: Game unit type, e.g SCV
        :return: list of units with the specific unit type.
        """
        return [unit for unit in self.worker_units if unit.get_unit_type() == unit_type] + \
               [unit for unit in self.military_units if unit.get_unit_type() == unit_type]

    def get_free_unit_of_type(self, unit_type):
        return [unit for unit in self.worker_units if unit.get_unit_type() == unit_type and unit.is_free()] + \
               [unit for unit in self.military_units if unit.get_unit_type() == unit_type and unit.is_free()]

    def on_step(self, latest_units_list):
        """
        Updates all units states accordingly with our data-structures.
        In short terms, removing dead units and adding new.
        """
        # Remove dead worker units
        self.update_dead_units(self.worker_units)

        # Remove dead scout units
        self.update_dead_units(self.scout_units)

        # Remove dead military units
        self.update_dead_units(self.military_units)

        # Update our workers
        self.add_new_units(latest_units_list, self.worker_units, self.is_worker_type, WorkerUnit)

        # Update our military units
        self.add_new_units(latest_units_list, self.military_units, self.is_military_type, MilitaryUnit)

        # update visible enemies
        self.visible_enemies = [unit for unit in latest_units_list if
                                unit.player == PLAYER_ENEMY and unit.unit_type.is_combat_unit]

        self.update_military_units()

        self.marauder_q_table.on_step()
        self.marine_q_table.on_step()
        self.cyclone_q_table.on_step()
        self.helion_q_table.on_step()

    def update_military_units(self):
        unit: MilitaryUnit
        for unit in self.military_units:
            e_in_sight = []
            e_that_can_attack = []  # enemies
            a_in_sight = []  # allies
            e_in_range = []
            for enemy in self.visible_enemies:
                distance = unit.get_distance_to(enemy)

                if math.floor(distance) <= enemy.unit_type.attack_range:
                    e_that_can_attack.append(enemy)
                if math.floor(distance) <= unit.sight_range:
                    e_in_sight.append(enemy)
                if distance <= unit.attack_range:
                    e_in_range.append(enemy)
            for ally in self.military_units:
                distance = unit.get_distance_to(ally.get_unit())
                if distance <= unit.sight_range and ally != unit:
                    a_in_sight.append(ally)

            unit.on_step(e_in_sight, e_that_can_attack, a_in_sight, e_in_range)

    def add_new_units(self, latest_units_list, known_units, unit_type_checker, unit_class):
        for latest_unit in latest_units_list:
            # If unit is of requested type
            if latest_unit.player == PLAYER_SELF and unit_type_checker(latest_unit):
                # Check if unit is not already in our list
                if not any(latest_unit.id == unit.get_id() for unit in known_units):
                    if latest_unit.unit_type.unit_typeid == UNIT_TYPEID.TERRAN_MARAUDER:
                        known_units.append(unit_class(latest_unit, self.idabot, self.marauder_q_table))
                    elif latest_unit.unit_type.unit_typeid == UNIT_TYPEID.TERRAN_MARINE:
                        known_units.append(unit_class(latest_unit, self.idabot, self.marine_q_table))
                    elif latest_unit.unit_type.unit_typeid == UNIT_TYPEID.TERRAN_SCV:
                        known_units.append(unit_class(latest_unit, self.idabot))
                    elif latest_unit.unit_type.unit_typeid == UNIT_TYPEID.TERRAN_CYCLONE:
                        known_units.append(unit_class(latest_unit,self.idabot, self.cyclone_q_table))
                    elif latest_unit.unit_type.unit_typeid == UNIT_TYPEID.TERRAN_HELLION:
                        known_units.append(unit_class(latest_unit,self.idabot, self.helion_q_table))
    def update_dead_units(self, unit_list):
        '''
        Removes all units of given list that are not alive anymore
        :param unit_list: List of abstraction units
        '''
        for current_unit in unit_list:
            if not current_unit.is_alive(): # TODO: är denna true när en enhet är i ett refinary?
                # current_unit.die()
                if current_unit is ScoutUnit:
                    # Mark the goal is visited if the unit died, prevents suicide mission next
                    self.idabot.scout_manager.visited.append(current_unit.get_goal())
                    self.idabot.scout_manager.frame_stamps.append(self.idabot.current_frame)
                    self.idabot.scout_manager.goals.remove(current_unit.get_goal())
                if current_unit.get_unit().unit_type.is_combat_unit:
                    print("TOTAL REWARD:", current_unit.total_reward)
                unit_list.remove(current_unit)

    def create_coalition(self, nr_coalitions):
        '''
        Tell csg to generate new coalitions from scratch.
        :param nr_coalitions: How many coalitions to divide units into
        :return: None, update internal state for coalitions (self.cs)
        '''
        # TODO: change this to something reasonable, change to military_unit
        info = {}
        info["militaryUnits"] = {
            military_type.get_unit_type(): len(self.get_units_of_type(military_type.get_unit_type()))
            for military_type in self.military_units
            if "militaryUnits" not in info
               or military_type.get_unit_type() not in info["militaryUnits"]
        }

        self.groups = self.csg.create_coalition(info["militaryUnits"], nr_coalitions)

    def add_units_to_coalition(self):
        units_to_add = []
        units_in_groups = []
        for group in self.groups:
            units_in_groups += group

        for unit in self.military_units:
            if unit not in units_in_groups:
                self.csg.add_unit(unit, self.groups)


    def is_military_type(self, unit):
        '''
        :param unit: Game unit type
        :return: If given unit is any military unit type
        '''
        return any(unit.unit_type == unit_type for unit_type in self.MILITARY_TYPES)

    def is_worker_type(self, unit):
        '''
        :param unit: Game unit type
        :return:  If given unit is any worker unit type
        '''
        return any(unit.unit_type == unit_type for unit_type in self.WORKER_TYPES)

    def command_group(self, task, group):
        for unit in group:
            if not unit.is_in_combat():
                unit.attack_move(task.pos)

    def command_unit(self, unit, task):
        """

        HÄR SKA VI SE TILL SÅ ATT SAKER HÄNDER.

        unit kan vara byggnad worker, eller grupp av military units. Man kan köra task.task_type för att se vilken typ av task det här

        Om det unit är byggnad och task train_unit finns det ingen garanti för att byggnaden kan producera uniten, får kolla med can_produce
        """
        print("Commanding unit: ", unit.get_id(), "to do task", task.task_type)


        if task.task_type is TaskType.SCOUT:
            self.idabot.scout_manager.scouts_requested -= 1
            self.scout_units.append(ScoutUnit(unit.unit, self.idabot.scout_manager, self.idabot.strategy_network,
                                              len(self.scout_units)))

        elif task.task_type is TaskType.MINING:
            minerals = self.idabot.get_mineral_fields(task.base_location)
            unit.set_mining(minerals[random.randint(0, len(minerals)-1)])

        elif task.task_type is TaskType.GAS:
            refineries = self.idabot.building_manager.get_buildings_of_type(UnitType(UNIT_TYPEID.TERRAN_REFINERY, self.idabot))
            for refinery in refineries:
                if refinery.get_pos().x == task.pos.x and refinery.get_pos().y == task.pos.y:
                    unit.set_gassing(refinery.get_unit())

        elif task.task_type is TaskType.BUILD:
            self.idabot.resource_manager.use(task.construct_building)
            unit.build(task.construct_building, task.pos)
