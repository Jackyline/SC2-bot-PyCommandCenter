import os

from typing import Optional
from library import *
from classes.resource_manager import ResourceManager
from classes.unit_manager import UnitManager


class MyAgent(IDABot):
    def __init__(self):
        IDABot.__init__(self)


    def on_game_start(self):
        IDABot.on_game_start(self)


    def on_step(self):
        IDABot.on_step(self)
        units = self.get_all_units()
        my_units = []
        enemy_units = []
        for unit in units:
            if unit.unit_type.unit_typeid == UNIT_TYPEID.TERRAN_MARINE:
                if unit.player == PLAYER_ENEMY:
                    enemy_units.append(unit)
                else:
                    my_units.append(unit)

        if len(enemy_units) > 0:
            for my in my_units:
                my.attack_unit(enemy_units[0])
        else:
            enemy_base = self.base_location_manager.get_player_starting_base_location(PLAYER_ENEMY)
            for unit in my_units:
                unit.move(enemy_base.position)





def main():
    coordinator = Coordinator(r"D:\StarCraft II\Versions\Base69232\SC2_x64.exe")

    bot1 = MyAgent()
    bot2 = MyAgent()

    participant_1 = create_participants(Race.Terran, bot1)
    participant_2 = create_participants(Race.Terran, bot2)
    #participant_2 = create_computer(Race.Terran, Difficulty.Easy)

    coordinator.set_real_time(False)
    coordinator.set_participants([participant_1, participant_2])
    coordinator.launch_starcraft()

    path = os.path.join(os.getcwd(), "maps","test6.SC2Map")
    coordinator.start_game(path)

    while coordinator.update():
        pass


if __name__ == "__main__":
    main()