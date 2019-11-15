import os

from typing import Optional
from library import *
from classes.resource_manager import ResourceManager
from classes.unit_manager import UnitManager
from classes.scouting_manager import ScoutingManger


class MyAgent(IDABot):
    def __init__(self):
        IDABot.__init__(self)
        self.resource_manager = ResourceManager(self.minerals, self.gas, self.current_supply, self)
        self.unit_manager = UnitManager(self)
        self.scout_manager = ScoutingManger(self)

    def on_game_start(self):
        IDABot.on_game_start(self)

    def on_step(self):
        IDABot.on_step(self)
        self.resource_manager.sync()
        self.unit_manager.on_step(self.get_my_units())
        self.scout_manager.on_step(self.get_my_units(), self.map_tools.width, self.map_tools.height)
        self.print_debug()
        self.map_tools.draw_text_screen(percentage_x=0.01, percentage_y=0.01, text=self.scout_manager.print_debug())

    def print_debug(self):
        objects = []

        # Get all of our units
        objects += self.get_my_units()

        # Get minerals that are close by to our starting base
        base = self.base_location_manager.get_player_starting_base_location(player_constant=PLAYER_SELF)
        objects += base.mineral_fields

        for unit in list(enumerate(sorted(objects, key=lambda u: u.id))):
            self.map_tools.draw_text(position=unit[1].position, text="{unittype} id: {id} position: {enumeratingIndex}".
                                     format(unittype=unit[1].unit_type,
                                            id=unit[1].id,
                                            enumeratingIndex=unit[1].position))


def main():
    coordinator = Coordinator(r"C:\Users\hanne\Desktop\StarCraft II\Versions\Base69232\SC2_x64.exe")

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
