import os

from typing import Optional
from library import *
from classes.resource_manager import ResourceManager
from classes.unit_manager import UnitManager
from classes.scouting_manager import ScoutingManager
from classes.print_debug import PrintDebug
from classes.building_manager import BuildingManager
from classes.q_agent import QAgent
from classes.stupid_agent import StupidAgent
from classes.q_table import QTable

class MyAgent(IDABot):
    def __init__(self):
        IDABot.__init__(self)
        self.resource_manager = ResourceManager(self.minerals, self.gas, self.current_supply, self)
        self.unit_manager = UnitManager(self)
        self.scout_manager = ScoutingManager(self)
        self.building_manager = BuildingManager(self)
        self.print_debug = PrintDebug(self, self.building_manager, self.unit_manager, self.scout_manager, True)
    def on_game_start(self):
        IDABot.on_game_start(self)

    def on_step(self):
        IDABot.on_step(self)
        self.resource_manager.sync()
        self.unit_manager.on_step(self.get_my_units())
        self.scout_manager.on_step(self.get_my_units(), self.map_tools.width, self.map_tools.height)
        self.building_manager.on_step(self.get_my_units())
        self.print_debug.on_step()



def main():
    coordinator = Coordinator(r"D:\StarCraft II\Versions\Base69232\SC2_x64.exe")

    bot1 = QAgent()
    bot2 = StupidAgent()

    participant_1 = create_participants(Race.Terran, bot2)
    participant_2 = create_participants(Race.Terran, bot1)
    #participant_2 = create_computer(Race.Random, Difficulty.Easy)

    coordinator.set_real_time(False)
    coordinator.set_participants([participant_1, participant_2])
    coordinator.launch_starcraft()

    path = os.path.join(os.getcwd(), "maps", "8-marines-vs-3-stalkers.SC2Map")
    coordinator.start_game(path)

    while coordinator.update():
        pass


if __name__ == "__main__":
    main()
