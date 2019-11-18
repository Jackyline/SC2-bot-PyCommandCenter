from classes.military_unit import MilitaryUnit
from classes.worker_unit import WorkerUnit
from classes.q_table import QTable
from library import *


class UnitManager:

    def __init__(self, idabot):
        self.idabot = idabot

        # All military types
        self.MILITARY_TYPES = [UnitType(UNIT_TYPEID.TERRAN_MARINE, idabot),
                               UnitType(UNIT_TYPEID.TERRAN_MARAUDER, idabot),
                               UnitType(UNIT_TYPEID.TERRAN_REAPER, idabot), UnitType(UNIT_TYPEID.TERRAN_GHOST, idabot),
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

        # List of our abstracted military units
        self.military_units = []

        #list of visible military units
        self.visible_enemies = []

        self.q_table = QTable(self.idabot)

    def get_info(self):
        '''
        :return: Info about all units and their type/workstation
        '''
        info = {}

        # {job_type : amount doing that job type}
        info["workerUnits"] = {worker.get_job().unit_type if worker.get_job() else None:
                                   len(self.get_units_working_on_type(worker.get_job()))
                               for worker in self.worker_units
                               if "workerUnits" not in info
                               or worker.get_job() not in info["workerUnits"]
                               }

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

        # Remove dead military units
        self.update_dead_units(self.military_units)

        # Update our workers
        self.add_new_units(latest_units_list, self.worker_units, self.is_worker_type, WorkerUnit)

        # Update our military units
        self.add_new_units(latest_units_list, self.military_units, self.is_military_type, MilitaryUnit)

        #update visible enemies
        self.visible_enemies = [unit for unit in latest_units_list if unit.player == PLAYER_ENEMY and unit.unit_type.is_combat_unit]

        self.update_military_units()
        self.q_table.on_step()


    def update_military_units(self):
        unit : MilitaryUnit
        for unit in self.military_units:
            e_in_range = []  # enemies
            a_in_range = []  # allies
            for enemy in self.visible_enemies:
                if self.idabot.map_tools.get_ground_distance(unit.get_unit().position, enemy.position) <= \
                        unit.get_unit_type().sight_range:
                    e_in_range.append(enemy)

            for ally in self.military_units:
                if self.idabot.map_tools.get_ground_distance(unit.get_unit().position, ally.get_unit().position) <= \
                        unit.get_unit_type().sight_range and ally != unit:
                    a_in_range.append(ally)



            unit.on_step(e_in_range, a_in_range)


    def add_new_units(self, latest_units_list, known_units, unit_type_checker, unit_class):
        for latest_unit in latest_units_list:
            # If unit is of requested type
            if latest_unit.player == PLAYER_SELF and unit_type_checker(latest_unit):
                # Check if unit is not already in our list
                if not any(latest_unit.id == unit.get_id() for unit in known_units):
                    known_units.append(unit_class(latest_unit, self.idabot, self.q_table))

    def update_dead_units(self, unit_list):
        '''
        Removes all units of given list that are not alive anymore
        :param unit_list: List of abstraction units
        '''
        for current_unit in unit_list:
            if not current_unit.is_alive():
                # current_unit.die()
                unit_list.remove(current_unit)

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
