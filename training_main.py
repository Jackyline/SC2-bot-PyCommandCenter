import os

from typing import Optional
from library import *
from classes.resource_manager import ResourceManager
from classes.unit_manager import UnitManager
from classes.assignment_manager import AssignmentManager
from classes.building_manager import BuildingManager
#from classes.task_manager import TaskManager
from strategy.strategy import Strategy
from classes.scouting_manager import ScoutingManager
from classes.print_debug import PrintDebug
from classes.building_manager import BuildingManager
from classes.stupid_agent import StupidAgent
# TODO:
# Add building strategy back again when torch installation finished
#from classes.building_strategy import BuildingStrategy


class MyAgent(IDABot):
    def __init__(self):
        IDABot.__init__(self)
        #self.building_manager = BuildingManager(self)
        #self.resource_manager = ResourceManager(self.minerals, self.gas, self.current_supply, self)
        self.unit_manager = UnitManager(self)
        #self.strategy_network = Strategy()
        #self.assignment_manager = AssignmentManager(unit_manager=self.unit_manager,
        #                                            building_manager=self.building_manager)
        #self.task_manager = TaskManager(self.assignment_manager)

        #self.scout_manager = ScoutingManager(self)
        #self.building_manager = BuildingManager(self)
        # TODO:
        # Add building strategy back again when torch installation finished
        #self.building_strategy = BuildingStrategy()
        #self.print_debug = PrintDebug(self, self.building_manager, self.unit_manager, self.scout_manager, True)

    def on_game_start(self):
        IDABot.on_game_start(self)

    def on_step(self):
        IDABot.on_step(self)
        #self.resource_manager.sync()

        # TODO: will be used from building_manager instead
        #command_center_types = [UnitType(UNIT_TYPEID.TERRAN_COMMANDCENTER, self),
        #                        UnitType(UNIT_TYPEID.TERRAN_COMMANDCENTERFLYING, self)]
        #command_centers = [b for b in self.get_my_units() if b.unit_type in command_center_types]

        # TODO: Is this how you get the actual seconds?  svar nej
        #curr_seconds = self.current_frame // 24
        # Minutes, Seconds
        #curr_time = int((curr_seconds) // 60) + (curr_seconds % 60) / 60
        #self.strategy = self.strategy_network.get_strategy([
        #   len(self.unit_manager.worker_units),
        #  len(self.unit_manager.military_units),
        #    self.resource_manager.resources.minerals,
        #    self.resource_manager.resources.gas,
        #    len(command_centers),
        #    curr_time
        #])

        self.unit_manager.on_step(self.get_all_units())
        #self.scout_manager.on_step()
        #self.building_manager.on_step(self.get_my_units())
        #self.print_debug.on_step()
        #self.assignment_manager.on_step()
        #self.task_manager.on_step()




def main():
    coordinator = Coordinator(r"C:\New starcraft\StarCraft II\Versions\Base69232\SC2_x64.exe")

    bot1 = StupidAgent()
    bot2 = MyAgent()

    participant_1 = create_participants(Race.Terran, bot1)
    participant_2 = create_participants(Race.Terran, bot2)
    #participant_2 = create_computer(Race.Random, Difficulty.Easy)

    coordinator.set_real_time(False)
    coordinator.set_participants([participant_1, participant_2])
    coordinator.launch_starcraft()

    path = os.path.join(os.getcwd(), "maps", "helion-marine-5-5-random-spawn.SC2Map")
    #path = os.path.join(os.getcwd(), "maps", "marine-marine-10-10-random-spawn.SC2Map")

    #path = os.path.join(os.getcwd(), "maps", "marauder-marine-random-spawn-concussive.SC2Map")
    #path = os.path.join(os.getcwd(), "maps", "marauder-marine-5-5-random-spawn-concussive.SC2Map")
    #path = os.path.join(os.getcwd(), "maps", "marauder-marine-5-10-random-spawn-concussive.SC2Map")
    #path = os.path.join(os.getcwd(), "maps", "marauder-marine-10-10-random-spawn-concussive.SC2Map")

    #path = os.path.join(os.getcwd(), "maps", "marauder-marauder-10-10-random-spawn-concussive.SC2Map")

    #path = os.path.join(os.getcwd(), "maps", "marauder-cyclone-10-10-random-spawn-concussive.SC2Map")

    #path = os.path.join(os.getcwd(), "maps", "cyclone-marine-10-10-random-spawn.SC2Map")
    coordinator.start_game(path)

    while coordinator.update():
        pass


if __name__ == "__main__":
    main()
