from library import UnitType, UNIT_TYPEID


class UnitManager:

    def __init__(self):
        # All military types
        self.MILITARY_TYPES = [UnitType(UNIT_TYPEID.TERRAN_MARINE, self), UnitType(UNIT_TYPEID.TERRAN_MARAUDER, self),
                               UnitType(UNIT_TYPEID.TERRAN_REAPER, self), UnitType(UNIT_TYPEID.TERRAN_GHOST, self),
                               UnitType(UNIT_TYPEID.TERRAN_HELLION, self), UnitType(UNIT_TYPEID.TERRAN_SIEGETANK, self),
                               UnitType(UNIT_TYPEID.TERRAN_CYCLONE, self), UnitType(UNIT_TYPEID.TERRAN_WIDOWMINE, self),
                               UnitType(UNIT_TYPEID.TERRAN_THOR, self),
                               UnitType(UNIT_TYPEID.TERRAN_VIKINGASSAULT, self),
                               UnitType(UNIT_TYPEID.TERRAN_VIKINGFIGHTER, self),
                               UnitType(UNIT_TYPEID.TERRAN_MEDIVAC, self),
                               UnitType(UNIT_TYPEID.TERRAN_LIBERATOR, self), UnitType(UNIT_TYPEID.TERRAN_RAVEN, self),
                               UnitType(UNIT_TYPEID.TERRAN_BANSHEE, self),
                               UnitType(UNIT_TYPEID.TERRAN_BATTLECRUISER, self),
                               UnitType(UNIT_TYPEID.TERRAN_AUTOTURRET, self)]
        # Worker types
        self.WORKER_TYPES = [UnitType(UNIT_TYPEID.TERRAN_SCV, self)]

        # List of our abstracted worker units
        self.worker_units = []

        # List of our abstracted military units
        self.military_units = []

    def get_unit_of_type(self, unit_type):
        # TODO implement this
        pass

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
        self.add_new_units(latest_units_list, self.worker_units, self.is_worker_type, WorkerUnit())

        # Update our military units
        self.add_new_units(latest_units_list, self.military_units, self.is_military_type, MilitaryUnit())

    def add_new_units(self, latest_units_list, known_units, unit_type_checker, unit_class):
        for latest_unit in latest_units_list:
            # If unit is of requested type
            if unit_type_checker(latest_unit):
                # Check if unit is not already in our list
                if not any(latest_unit.id == unit.id for unit in known_units):
                    known_units.append(unit_class(latest_unit))  # TODO append new worker

    def update_dead_units(self, unit_list):
        '''
        Removes all units of given list that are not alive anymore
        :param unit_list: List of abstraction units
        '''
        for current_unit in unit_list:
            if not current_unit.object.is_alive:
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


class WorkerUnit:
    # TODO make this into own file etc etc
    def __init__(self):
        pass


class MilitaryUnit:
    # TODO make this into own file etc etc
    def __init__(self):
        pass
