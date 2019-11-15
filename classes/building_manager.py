from library import *
from classes.building_unit import BuildingUnit
from collections import defaultdict


class BuildingManager:
    def __init__(self, IDABot : IDABot):
        self.IDABot = IDABot
        self.buildings = []                         #:[BuildingUnit]
        self.under_construction = []                      #:[BuildingUnit]


    def on_step(self, buildings : [Unit]):
        """ Update internal lists 'buildings' and 'under_construction'. """

        # Check for completed buildings in self.under_construction
        for b in self.under_construction:
            if b.get_unit().is_completed:
                self.under_construction.remove(b)
                self.buildings.append(b)

        #Add undiscovered buildings to self.buildings and self.under_construction
        for building in buildings:
            if not building in [b.get_unit() for b in self.buildings + self.under_construction]:
                # Building is not in self.buildings, add it
                building_unit = BuildingUnit(building)
                if not building.is_completed:
                    self.under_construction.append(building_unit)
                else:
                    self.buildings.append(building_unit)

        # Remove buildings that no longer exists i.e. are destroyed
        if len(buildings) != len(self.buildings) + len(self.under_construction):
            self.buildings = [b for b in self.buildings if b.get_unit() in buildings]
            self.under_construction = [b for b in self.under_construction if b.get_unit() in buildings]


    def get_buildings_of_type(self, type : UnitType):
        return [b for b in self.buildings if b.get_unit_type() == type]

    def get_under_construction_of_type(self, type : UnitType):
        return [b for b in self.under_construction if b.get_unit_type() ==  type]

    def train(self, building : BuildingUnit, unit : Unit): # Todo: ändra till klasserna som vi själva gjort för unit
        pass

    def print_debug(self):
        types_buildings = {}
        return_string = ""
        for b in self.buildings:
            types_buildings[b.get_unit_type_id()] = types_buildings.get(b.get_unit_type_id(), 0) +1
        for key, value in types_buildings.items():
            return_string += "{}: {}\n".format(key, value)

        return return_string
