import math
import os
import winsound

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
from classes.stupid_agent import StupidAgent

# Only handle the predicted strategy this often (seconds)
HANDLE_STRATEGY_DELAY = 5


class MyAgent(IDABot):
    def __init__(self):
        IDABot.__init__(self)
        self.workers = []
        self.building_manager = BuildingManager(self)
        self.resource_manager = ResourceManager(self.minerals, self.gas, self.current_supply, self)
        self.unit_manager = UnitManager(self)
        self.strategy_network = Strategy(self)
        self.assignment_manager = AssignmentManager(self)
        self.scout_manager = ScoutingManager(self)
        self.building_strategy = BuildingStrategy(self, self.resource_manager, self.assignment_manager)
        self.print_debug = PrintDebug(self, self.building_manager, self.unit_manager, self.scout_manager,
                                      self.building_strategy, True)
        self.first = True
        self.counter = 0
        self.left = True
        self.dance = 0
        # Last time that strategy was handled by generating tasks etc
        self.last_handled_strategy = 0

    def on_game_start(self):
        IDABot.on_game_start(self)

    def on_step(self):
        IDABot.on_step(self)
        if self.first:
            for worker in self.get_my_units():
                if worker.unit_type.is_worker:
                    self.workers.append(worker)
                    worker.move(Point2D(31, 31))
            winsound.PlaySound('stadiljus1.wav', winsound.SND_ASYNC | winsound.SND_ALIAS)
            self.first = False
        done = False

        if self.current_frame % 2 == 0:
            self.counter += 1
            for worker in self.workers:
                if not worker.is_idle:
                    done = False
                    break
                else:
                    done = True
            if done:


                if self.counter > 100:
                    self.counter = 0
                    if self.left:
                        self.left = False
                    else:
                        self.left = True

                if self.left:
                    self.workers = [self.workers[(i + 1) % len(self.workers)]
                                    for i, x in enumerate(self.workers)]
                else:
                    self.workers = [self.workers[(i - 1) % len(self.workers)]
                                    for i, x in enumerate(self.workers)]
                for i in range(len(self.workers)):
                    self.get_circle_coord(self.workers[i], 31, i)

    def get_circle_coord(self, worker, center, index):
        if 30 < self.counter < 70:
            radius = 4.0
        elif 70 < self.counter < 80:
            radius = 3.0
        else:
            radius = 6.0
        angle = 2 * math.pi / 12
        worker.move(Point2D(center + (radius * math.cos(angle * index)), center + (radius * math.sin(angle * index))))

    def rightRotate(self, lists, num):
        output_list = []

        # Will add values from n to the new list
        for item in range(len(lists) - num, len(lists)):
            output_list.append(lists[item])

            # Will add the values before
        # n to the end of new list
        for item in range(0, len(lists) - num):
            output_list.append(lists[item])

        self.workers = output_list


def main():
    coordinator = Coordinator(r"C:\Users\hanne\Desktop\StarCraft II\Versions\Base69232\SC2_x64.exe")

    bot1 = MyAgent()
    # bot2 = StupidAgent()

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
