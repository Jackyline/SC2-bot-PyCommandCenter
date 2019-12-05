import os

from typing import Optional
from library import *
from classes.resource_manager import ResourceManager
from classes.unit_manager import UnitManager
from classes.assignment_manager import AssignmentManager
from classes.building_manager import BuildingManager
from classes.task_manager import TaskManager
from strategy.strategy import Strategy
from classes.scouting_manager import ScoutingManager
from classes.print_debug import PrintDebug
from classes.building_manager import BuildingManager
from classes.building_strategy import BuildingStrategy
from strategy.training_data import ALL_BUILDINGS, UNIT_TYPES


class MyAgent(IDABot):
    def __init__(self):
        IDABot.__init__(self)
        self.building_manager = BuildingManager(self)
        self.resource_manager = ResourceManager(self.minerals, self.gas, self.current_supply, self)
        self.unit_manager = UnitManager(self)
        self.strategy_network = Strategy(self)
        self.assignment_manager = AssignmentManager(unit_manager=self.unit_manager,
                                                    building_manager=self.building_manager)
        self.task_manager = TaskManager(self.assignment_manager)

        self.scout_manager = ScoutingManager(self)
        self.building_manager = BuildingManager(self)
        self.building_strategy = BuildingStrategy(self.resource_manager)
        self.print_debug = PrintDebug(self, self.building_manager, self.unit_manager, self.scout_manager,
                                      self.building_strategy, self.strategy_network, True)

    def on_game_start(self):
        IDABot.on_game_start(self)

    def on_step(self):
        IDABot.on_step(self)
        self.resource_manager.sync()
        self.unit_manager.on_step(self.get_all_units())
        self.scout_manager.on_step()
        self.building_manager.on_step(self.get_my_units())
        self.print_debug.on_step()
        self.assignment_manager.on_step()
        self.task_manager.on_step()


def main():
    coordinator = Coordinator(r"C:\Users\Dylan\Desktop\StarCraft II\Versions\Base69232\SC2_x64.exe")

    bot1 = MyAgent()
    # bot2 = MyAgent()

    participant_1 = create_participants(Race.Terran, bot1)
    # participant_2 = create_participants(Race.Terran, bot2)
    participant_2 = create_computer(Race.Random, Difficulty.Easy)

    coordinator.set_real_time(False)
    coordinator.set_participants([participant_1, participant_2])
    coordinator.launch_starcraft()

    path = os.path.join(os.getcwd(), "maps", "InterloperTest.SC2Map")
    coordinator.start_game(path)

    while coordinator.update():
        pass


if __name__ == "__main__":
    main()
