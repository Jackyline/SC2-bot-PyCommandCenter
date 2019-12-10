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
from strategy.strategy import StrategyName
from classes.task import Task, TaskType

# Only handle the predicted strategy this often (seconds)
HANDLE_STRATEGY_DELAY = 5


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
                                      self.building_strategy, True)

        # Last time that strategy was handled by generating tasks etc
        self.last_handled_strategy = 0

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

        # Generate jobs depending on strategy
        self.handle_strategy()

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

    def handle_strategy(self):
        """
        Generates jobs depending on our chosen strategy
        """

        # Calculate new predicted strategy
        strategy = self.strategy_network.get_strategy()

        curr_seconds = self.current_frame // 24

        # Only look at new strategy and generate new tasks every now and then
        if curr_seconds - self.last_handled_strategy < HANDLE_STRATEGY_DELAY:
            return

        # Now handling a strategy decision
        self.last_handled_strategy = curr_seconds



        # Get all of our command centers
        command_centers = self.building_manager.get_buildings_of_type(UnitType(UNIT_TYPEID.TERRAN_COMMANDCENTER, self))

        if strategy == StrategyName.OFFENSIVE:
            offensive_groups = 4
            defensive_groups = 1
            attack_pos = self.scout_manager.get_enemy_target()
        else:  # strategy == StrategyName.DEFENSIVE
            offensive_groups = 0
            defensive_groups = len(command_centers)

        # Generate all offensive tasks
        offensive_tasks = [Task(task_type=TaskType.ATTACK,
                                pos=attack_pos)
                           for i in range(offensive_groups)]

        # Generate all defensive tasks
        defensive_tasks = [Task(task_type=TaskType.DEFEND,
                                pos=command_centers[i % len(command_centers)])
                           # Loop through all bases we have and
                           for i in range(defensive_groups) if command_centers]

        # Add all generated tasks to assignment_manager
        for task in [*offensive_tasks, *defensive_tasks]:
            self.assignment_manager.add_task(task)


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
