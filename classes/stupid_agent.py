
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
        my = []
        enemy = []
        for unit in self.get_all_units():
            if unit.player == PLAYER_SELF:
                my.append(unit)
            else:
                enemy.append(unit)

        for unit in my:
            unit.attack_unit(random.choice(enemy))
        """

