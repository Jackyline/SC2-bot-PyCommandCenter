import os

from typing import Optional
from library import *
from classes.resource_manager import ResourceManager
from classes.unit_manager import UnitManager
from classes.task_manager import TaskManager
from classes.building_manager import BuildingManager
from classes.task_generator import TaskGenerator


class MyAgent(IDABot):
    def __init__(self):
        IDABot.__init__(self)
        self.resource_manager = ResourceManager(self.minerals, self.gas, self.current_supply, self)
        self.unit_manager = UnitManager(self)
        self.building_manager = BuildingManager(self)
        self.task_manager = TaskManager(unit_manager=self.unit_manager, building_manager=self.building_manager)
        self.task_generator = TaskGenerator(self.task_manager)

    def on_game_start(self):
        IDABot.on_game_start(self)

    def on_step(self):
        IDABot.on_step(self)
        self.resource_manager.sync()
        self.unit_manager.on_step(self.get_my_units())
        self.task_generator.on_step()
        self.task_manager.on_step()

def main():
    coordinator = Coordinator(r"D:\StarCraft II\Versions\Base69232\SC2_x64.exe")

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