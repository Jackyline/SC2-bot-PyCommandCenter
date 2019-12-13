from library import *
from classes.building_unit import BuildingUnit
from collections import defaultdict
from classes.task_type import TaskType
from classes.task import Task

class BuildingManager:
    def __init__(self, ida_bot : IDABot):
        self.ida_bot = ida_bot
        self.buildings = []                         #:[BuildingUnit]
        self.under_construction = []                      #:[BuildingUnit]


    def on_step(self, buildings : [Unit]):
        """ Update internal lists 'buildings' and 'under_construction'. """

        # Check for completed buildings in self.under_construction
        buildings = [unit for unit in buildings if unit.unit_type.is_building or unit.unit_type.is_addon]
        for b in self.under_construction:
            if b.get_unit().is_completed:
                self.under_construction.remove(b)
                self.buildings.append(b)

        #Add undiscovered buildings to self.buildings and self.under_construction
        for building in buildings:
            if not building in [b.get_unit() for b in self.buildings + self.under_construction]:
                # Building is not in self.buildings, add it

                building_unit = BuildingUnit(building)
                if building.unit_type.is_addon:
                    closest = self.find_closest_building(building_unit)
                    closest.has_techlab = True
                if not building.is_completed:
                    self.under_construction.append(building_unit)
                else:
                    self.buildings.append(building_unit)
            if building.unit_type.unit_typeid == UNIT_TYPEID.TERRAN_BARRACKSTECHLAB and not building.is_being_constructed:
                if not self.ida_bot.unit_manager.concussive_shells:
                    building.ability(ABILITY_ID.RESEARCH_CONCUSSIVESHELLS)
                    self.ida_bot.unit_manager.concussive_shells = True
                elif not self.ida_bot.unit_manager.combat_shield:
                    building.ability(ABILITY_ID.RESEARCH_COMBATSHIELD)
                    self.ida_bot.unit_manager.combat_shield = True
            #elif building.unit_type.unit_typeid == UNIT_TYPEID.TERRAN_SUPPLYDEPOT and not building.is_being_constructed:
            #    building.ability(ABILITY_ID.MORPH_SUPPLYDEPOT_LOWER)

        # Remove buildings that no longer exists i.e. are destroyed
        if len(buildings) != len(self.buildings) + len(self.under_construction):
            self.buildings = [b for b in self.buildings if b.get_unit() in buildings]
            self.under_construction = [b for b in self.under_construction if b.get_unit() in buildings]

    def find_closest_building(self, building):
        closest_building = self.buildings[0]
        closest_distance = self.ida_bot.get_distance_to(closest_building.get_unit().position, building.get_unit().position)

        for building2 in [unit for unit in self.buildings if not unit.get_unit().unit_type.is_addon]:
            distance = self.ida_bot.get_distance_to(building.get_unit().position, building2.get_unit().position)
            if distance < closest_distance:
                closest_building = building2
                closest_distance = distance
        return closest_building

    def get_geysers(self, base_location: BaseLocation):
        """
        Det finns någon bugg som gör det krångligt att hitta geysrar. Denna funktion gavs på labbsidan
        """
        geysers = []
        for geyser in base_location.geysers:
            for unit in self.ida_bot.get_all_units():
                if unit.unit_type.is_geyser \
                        and geyser.tile_position.x == unit.tile_position.x \
                        and geyser.tile_position.y == unit.tile_position.y:
                    geysers.append(unit)
        return geysers

    def get_buildings_of_type(self, type : UnitType):
        return [b for b in self.buildings if b.get_unit_type() == type]

    def get_under_construction_of_type(self, type : UnitType):
        return [b for b in self.under_construction if b.get_unit_type() ==  type]

    def get_total_buildings_of_type(self, type: UnitType):
        return [b for b in self.buildings + self.under_construction if b.get_unit_type() == type]

    def get_my_producers(self, unit_type: UnitType):
        """ Returns a list of units which can build or train units of type unit_type """
        producers = []
        type_data = self.ida_bot.tech_tree.get_data(unit_type)
        what_builds = type_data.what_builds

        for unit in self.ida_bot.get_my_units():
            if unit.unit_type in what_builds:
                producers.append(unit)

        return producers


    def get_refinery(self, geyser: Unit):
        """ Returns: A refinery which is on top of unit `geyser` if any, None otherwise """

        def squared_distance(p1: Point2D, p2: Point2D) -> float:
            return (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2

        for unit in self.ida_bot.get_my_units():
            if unit.unit_type.is_refinery and squared_distance(unit.position, geyser.position) < 1:
                return unit

        return None

    def command_building(self, building : BuildingUnit, task):
        if task.task_type is TaskType.ADD_ON:
            for building in self.get_my_producers(task.construct_building):
                if building.is_completed:
                    building_tmp = BuildingUnit(building)
                    self.ida_bot.resource_manager.use(task.construct_building)
                    building_tmp.train(task.construct_building)
                    building_tmp.set_task(task)

        elif task.task_type is TaskType.TRAIN:
            if task.produce_unit.unit_typeid == UNIT_TYPEID.TERRAN_SCV and len(self.ida_bot.unit_manager.worker_units) > 50:
                return
            if building.get_unit() in self.get_my_producers(task.produce_unit):
                building.train(task.produce_unit)
                building.set_task(task)

    def print_debug(self):
        types_buildings = {}
        return_string = ""
        for b in self.buildings:
            types_buildings[b.get_unit_type_id()] = types_buildings.get(b.get_unit_type_id(), 0) +1
        for key, value in types_buildings.items():
            return_string += "{}: {}\n".format(key, value)

        return return_string
