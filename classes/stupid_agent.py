
from library import *
import random
import math
import numpy as np
from classes.building_placer import BuildingPlacer
class StupidAgent2(IDABot):
    def __init__(self):
        IDABot.__init__(self)

    def on_game_start(self):
        IDABot.on_game_start(self)

    def on_step(self):
        IDABot.on_step(self)

        scvs = [unit for unit in self.get_my_units() if unit.unit_type.unit_typeid == UNIT_TYPEID.TERRAN_SCV and unit.is_idle]
        baselocation_manager : BaseLocationManager = self.base_location_manager
        baselocations = baselocation_manager.get_occupied_base_locations(PLAYER_SELF)
        index = 0
        for baselocation in baselocations:
            for geyser in baselocation.geysers:
                if self.get_refinery(geyser) is None:
                    scvs[index].build_target(UnitType(UNIT_TYPEID.TERRAN_REFINERY, self), geyser)
                    index += 1

    def get_refinery(self, geyser: Unit):
        """ Returns: A refinery which is on top of unit `geyser` if any, None otherwise """

        def squared_distance(p1: Point2D, p2: Point2D) -> float:
            return (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2

        for unit in self.get_my_units():
            if unit.unit_type.is_refinery and squared_distance(unit.position, geyser.position) < 1:
                return unit

        return None


class StupidAgent(IDABot):
    def __init__(self):
        IDABot.__init__(self)
        self.buildings = []
        self.supply_depots = []
        self.unit_builders = []
        self.rest_buildings = []
        #self.our_building_placer = BuildingPlacer(self.start_location)


    def on_game_start(self):
        IDABot.on_game_start(self)
        if self.start_location.x == 125.5 and self.start_location.y == 30.5:
            self.file = "resources/bottom_right/"
        else:
            self.file = "resources/top_left/"
    def on_step(self):
        IDABot.on_step(self)

        for unit in self.get_my_units():
            if unit in self.buildings or not unit.unit_type.is_building or unit.unit_type.unit_typeid == UNIT_TYPEID.TERRAN_COMMANDCENTER:
                continue
            self.buildings.append(unit)
            id = unit.unit_type.unit_typeid
            if id == UNIT_TYPEID.TERRAN_SUPPLYDEPOT:
                self.supply_depots.append((unit.position.x, unit.position.y))
                np.save(self.file+"supply_depot_pos.npy", self.supply_depots)
            elif id == UNIT_TYPEID.TERRAN_BARRACKS or id == UNIT_TYPEID.TERRAN_STARPORT or id == UNIT_TYPEID.TERRAN_FACTORY:
                self.unit_builders.append((unit.position.x, unit.position.y))
                np.save(self.file+"unit_builders_pos.npy", self.unit_builders)
            elif id == UNIT_TYPEID.TERRAN_ENGINEERINGBAY or id == UNIT_TYPEID.TERRAN_ARMORY:
                self.rest_buildings.append((unit.position.x, unit.position.y))
                np.save(self.file+"rest_buildings_pos.npy", self.rest_buildings)

        """
        for unit in self.get_all_units():
            unit.stop_dance()
        

        my = []
        enemy = []
        for unit in self.get_all_units():
            if unit.player == PLAYER_SELF and unit.unit_type.is_combat_unit:
                my.append(unit)
            elif unit.player == PLAYER_ENEMY and unit.unit_type.is_combat_unit:
                enemy.append(unit)
        a = 3



        for unit in my:
            if enemy:
                unit.attack_unit(closest_enemy(enemy,unit))

        """
        a=3
def get_distance_to(unit, enemy):
    """
    Return the distance to a unit, if units is none 10 will be returned
    :param unit:
    :return:
    """

    return math.sqrt((unit.position.x - enemy.position.x)**2 +
                             (unit.position.y - enemy.position.y)**2)



def closest_enemy(enemies, unit):
    closest_enemy = enemies[0]
    closest_distance = get_distance_to(unit,closest_enemy)
    for enemy in enemies:
        distance = get_distance_to(unit, enemy)
        if distance < closest_distance:
            closest_distance = distance
            closest_enemy = enemy
    return closest_enemy