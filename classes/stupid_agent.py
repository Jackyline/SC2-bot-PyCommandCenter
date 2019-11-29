
from library import *
import random

class StupidAgent(IDABot):
    def __init__(self):
        IDABot.__init__(self)


    def on_game_start(self):
        IDABot.on_game_start(self)

    def on_step(self):
        IDABot.on_step(self)
        """
        for unit in self.get_all_units():
            unit.stop_dance()
        """
        my = []
        enemy = []
        for unit in self.get_all_units():
            if unit.player == PLAYER_SELF and unit.unit_type.is_combat_unit:
                my.append(unit)
            elif unit.player == PLAYER_ENEMY and unit.unit_type.is_combat_unit:
                enemy.append(unit)

        for unit in my:
            if len(enemy) > 0 and not unit.has_target:
                unit.attack_unit(random.choice(enemy))

