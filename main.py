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
from classes.stupid_agent import StupidAgent, StupidAgent2
from classes.building_placer import BuildingPlacer
import random
# Only handle the predicted strategy this often (seconds)
HANDLE_STRATEGY_DELAY = 5


class MyAgent(IDABot):
    def __init__(self):
        IDABot.__init__(self)
        self.minerals_in_base = {}
        self.building_manager = BuildingManager(self)
        self.resource_manager = ResourceManager(self.minerals, self.gas, self.current_supply, self)
        self.unit_manager = UnitManager(self)
        self.strategy_network = Strategy(self)
        self.assignment_manager = AssignmentManager(self)
        self.scout_manager = ScoutingManager(self)
        self.building_strategy = BuildingStrategy(self, self.resource_manager, self.assignment_manager)
        self.print_debug = PrintDebug(self, self.building_manager, self.unit_manager, self.scout_manager,
                                      self.building_strategy, True)
        self.our_building_placer = None
        # Last time that strategy was handled by generating tasks etc
        self.last_handled_strategy = 0
        self.first_tick = True

        self.base_right = None
        self.choke_points_right = {(24.25, 28.5): Point2D(30, 60), (56.25, 130.5): Point2D(53, 118),
                                   (58.75, 99.0): Point2D(47, 92), (129.25, 54.5): Point2D(107, 64),
                                   (63.75, 51.0): Point2D(57, 73), (93.25, 69.0): Point2D(80, 81),
                                   (88.25, 117.0): Point2D(73, 115), (92.5, 143.5): Point2D(77, 134),
                                   (26.5, 137.5): Point2D(30, 133), (22.75, 113.5): Point2D(31, 118),
                                   (59.5, 24.5): Point2D(51, 36), (95.75, 37.5): Point2D(88, 49),
                                   (24.25, 83.5): Point2D(44, 95), (127.75, 139.5): Point2D(115, 135),
                                   (127.75, 84.5): Point2D(123, 98), (127.75, 28.5): Point2D(119, 47)}

        self.choke_points_left = {(58.75, 99.0): Point2D(58, 80), (24.25, 139.5): Point2D(33, 121),
                                  (127.75, 139.5): Point2D(124, 106), (92.5, 143.5): Point2D(100, 128),
                                  (56.25, 130.5): Point2D(64, 120), (22.75, 113.5): Point2D(45, 104),
                                  (127.75, 84.5): Point2D(113, 77), (24.25, 28.5): Point2D(37, 32),
                                  (24.25, 83.5): Point2D(29, 72), (88.25, 117.0): Point2D(96, 92),
                                  (93.25, 69.0): Point2D(108, 71), (59.5, 24.5): Point2D(71, 31),
                                  (95.75, 37.5): Point2D(98, 49), (63.75, 51.0): Point2D(79, 53),
                                  (129.25, 54.5): Point2D(121, 50), (127.75, 28.5): Point2D(117, 37)}

    def on_game_start(self):
        self.our_building_placer = BuildingPlacer(self.start_location, self)
        IDABot.on_game_start(self)

    def on_step(self):
        IDABot.on_step(self)

        # Sync minerals available only sometimes
        if self.current_frame % 100 == 0:
            self.minerals_in_base.clear()
            for base in self.base_location_manager.get_occupied_base_locations(PLAYER_SELF):
                self.minerals_in_base[base] = [mineral for mineral in self.get_mineral_fields(base)]

        # first sync units, buildings and resources
        self.resource_manager.sync()
        self.unit_manager.on_step(self.get_all_units())
        self.building_manager.on_step(self.get_my_units())
        self.building_strategy.action()

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

        curr_seconds = self.current_frame // 24

        # Only look at new strategy and generate new tasks every now and then
        if curr_seconds - self.last_handled_strategy < HANDLE_STRATEGY_DELAY:
            return

        # Calculate new predicted strategy
        strategy = self.strategy_network.get_strategy()

        # Now handling a strategy decision
        self.last_handled_strategy = curr_seconds

        # Get all of our command centers
        command_centers = self.building_manager.get_buildings_of_type(UnitType(UNIT_TYPEID.TERRAN_COMMANDCENTER, self)) + \
                          self.building_manager.get_under_construction_of_type(UnitType(UNIT_TYPEID.TERRAN_COMMANDCENTER, self))

        if strategy == StrategyName.OFFENSIVE:
            offensive_groups = 1
            defensive_groups = 0

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
                                pos=self.get_choke_point(command_centers[i % len(command_centers)].get_pos()))
                           # Loop through all bases we have and
                           for i in range(defensive_groups) if command_centers]

        # Add all generated tasks to assignment_manager
        for task in [*offensive_tasks, *defensive_tasks]:
            self.assignment_manager.add_task(task)

    def get_choke_point(self, position : Point2D):
        if self.base_right is None:
            self.base_right = self.base_location_manager.get_player_starting_base_location(PLAYER_SELF).position.x > 50

        # Choke points are based on base_location.position and :param: position is the command center position so
        # we choose the base_location position that is closest to our command center
        locations = list(map(lambda x: x.position, self.base_location_manager.get_occupied_base_locations(PLAYER_SELF)))
        closes_pos = None
        shortest_distance = 9999
        for loc in locations:
            dist = ((position.x - loc.x)**2 + (position.y - loc.y)**2)**0.5
            if dist < shortest_distance:
                closes_pos = loc
                shortest_distance = dist

        tPos = (closes_pos.x, closes_pos.y)
        return self.choke_points_right[tPos] if self.base_right else self.choke_points_left[tPos]


def main():
    coordinator = Coordinator(r"C:\New starcraft\StarCraft II\Versions\Base69232\SC2_x64.exe")

    bot1 = MyAgent()
    #bot2 = StupidAgent2()

    participant_1 = create_participants(Race.Terran, bot1)
    #participant_2 = create_participants(Race.Terran, bot2)
    participant_2 = create_computer(Race.Terran, Difficulty.Easy)

    coordinator.set_real_time(False)
    coordinator.set_participants([participant_1, participant_2])
    coordinator.launch_starcraft()

    path = os.path.join(os.getcwd(), "maps", "InterloperTest.SC2Map")
    coordinator.start_game(path)

    while coordinator.update():
        pass


if __name__ == "__main__":
    main()
