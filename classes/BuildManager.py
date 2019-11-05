from library import *

class BuildManager:
    def __init__(self, IDABot : IDABot):
        self.IDABot = IDABot
        self.buildings = []                         #:[BuildingUnit]
        self.constructing = []                      #:[BuildingUnit]


    def on_step(self, buildings : [Unit]):
        """ Add any new buildings to our internal list 'buildings' and remove any that have been destoryed. """

        for building in buildings:
            for b in self.buildings:
                if building == b.get_unit():
                    break
            else: # Building is not in self.buildings, add it
                # TODO: create BuildingUnit instance and append
                self.buildings.append(buildings)

        # If we have buildings that no longer exists i.e. are destroyed
        if len(buildings) != len(self.buildings):
            pass






        return

    def get_building(self, type : UnitType): -> BuildingUnit
        """ Returns all finished building of the specified type. """
        pass

    def get_constructing(self, type : UnitType):
        """ Returns all constructing buildings of the specified type """
        pass