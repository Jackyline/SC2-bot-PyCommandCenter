import numpy as np
from library import *
class BuildingPlacer():
    def __init__(self, start_location, ida_bot: IDABot):
        if start_location.x == 125.5 and start_location.y == 30.5:
            self.file = "resources/bottom_right/"
        else:
            self.file = "resources/top_left/"

        self.supply_depot_pos = list(np.load(self.file+"supply_depot_pos.npy"))
        self.unit_builders_pos = list(np.load(self.file+"unit_builders_pos.npy"))
        self.rest_buildings = list(np.load(self.file+"rest_buildings_pos.npy"))
        self.ida_bot = ida_bot

    def get_build_location(self, unit_type):
        id = unit_type.unit_typeid
        if not self.supply_depot_pos:
            self.supply_depot_pos = list(np.load(self.file+"supply_depot_pos.npy"))
        if not self.unit_builders_pos:
            self.unit_builders_pos = list(np.load(self.file+"unit_builders_pos.npy"))
        if not self.rest_buildings:
            self.rest_buildings = list(np.load(self.file+"rest_buildings_pos.npy"))
        if id == UNIT_TYPEID.TERRAN_SUPPLYDEPOT and len(self.supply_depot_pos) > 0:
            pos= self.supply_depot_pos.pop(0)
            return Point2D(pos[0], pos[1])
        elif (id == UNIT_TYPEID.TERRAN_BARRACKS or id == UNIT_TYPEID.TERRAN_STARPORT or id == UNIT_TYPEID.TERRAN_FACTORY) and len(self.unit_builders_pos) > 0:
            pos= self.unit_builders_pos.pop(0)
            return Point2D(pos[0], pos[1])
        elif (id == UNIT_TYPEID.TERRAN_ENGINEERINGBAY or id == UNIT_TYPEID.TERRAN_ARMORY) and len(self.rest_buildings) > 0:
            pos = self.rest_buildings.pop(0)
            return Point2D(pos[0], pos[1])
        else:
            building_placer : BuildingPlacer = self.ida_bot.building_placer
            return building_placer.get_build_location_near(
                self.ida_bot.base_location_manager.get_player_starting_base_location(PLAYER_SELF).depot_position, unit_type)