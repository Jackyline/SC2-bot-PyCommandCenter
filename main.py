import os

from typing import Optional
from library import *
from classes.resource_manager import ResourceManager
from classes.unit_manager import UnitManager
from classes.assignment_manager import AssignmentManager
from classes.building_manager import BuildingManager
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
        self.assignment_manager = AssignmentManager(self)
        self.scout_manager = ScoutingManager(self)
        self.building_strategy = BuildingStrategy(self, self.resource_manager, self.assignment_manager)
        self.print_debug = PrintDebug(self, self.building_manager, self.unit_manager, self.scout_manager,
                                      self.building_strategy, self.strategy_network, True)

    def on_game_start(self):
        IDABot.on_game_start(self)

    def on_step(self):
        IDABot.on_step(self)

        # first sync units, buildings and resources
        self.resource_manager.sync()
        self.unit_manager.on_step(self.get_all_units())
        self.building_manager.on_step(self.get_my_units())

        # then run specific AI parts
        self.scout_manager.on_step()
        self.assignment_manager.on_step()
        self.print_debug.on_step()

    def get_mineral_fields(self, base_location: BaseLocation):
        """ Given a base_location, this method will find and return a list of all mineral fields (Unit) for that base """
        mineral_fields = []
        for mineral_field in base_location.mineral_fields:
            for unit in self.get_all_units():
                if unit.unit_type.is_mineral \
                        and mineral_field.tile_position.x == unit.tile_position.x \
                        and mineral_field.tile_position.y == unit.tile_position.y:
                    mineral_fields.append(unit)
        return mineral_fields

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
