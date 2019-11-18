from library import *
from classes.resource_manager import ResourceManager
from classes.unit_manager import UnitManager
from classes.scouting_manager import ScoutingManager
from classes.print_debug import PrintDebug
from classes.building_manager import BuildingManager
from classes.military_unit import MilitaryUnit


class QAgent(IDABot):
    def __init__(self):
        IDABot.__init__(self)
        self.resource_manager = ResourceManager(self.minerals, self.gas, self.current_supply, self)
        self.unit_manager = UnitManager(self)
        self.scout_manager = ScoutingManager(self)
        self.building_manager = BuildingManager(self)
        self.print_debug = PrintDebug(self, self.building_manager, self.unit_manager, self.scout_manager, True)
        self.middle_of_map = None

    def on_game_start(self):
        IDABot.on_game_start(self)
        self.first_tick = True

    def on_step(self):
        IDABot.on_step(self)
        self.resource_manager.sync()
        self.unit_manager.on_step(self.get_all_units())
        #self.scout_manager.on_step(self.get_my_units(), self.map_tools.width, self.map_tools.height)
        self.building_manager.on_step(self.get_my_units())
        self.print_debug.on_step()

        if self.first_tick:
            self.first_tick= False
            self.middle_of_map = Point2D( self.map_tools.width/2,self.map_tools.height/2)
        """
        unit : MilitaryUnit
        for unit in self.unit_manager.get_units_of_type(UnitType(UNIT_TYPEID.TERRAN_MARINE, self)):
            if unit.is_in_combat():
                unit.on_step()
            else:
                pass
                #unit.get_unit().move(self.middle_of_map)

        #print(self.current_frame)
        """