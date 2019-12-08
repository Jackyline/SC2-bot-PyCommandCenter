
from library import *
import random
import math
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
        a = 3



        for unit in my:
            if enemy:
                unit.attack_unit(closest_enemy(enemy,unit))


def get_distance_to(unit, enemy):
    """
    Return the distance to a unit, if units is none 10 will be returned
    :param unit:
    :return:
    """

    return math.sqrt((unit.position.x - enemy.position.x)**2 +
                             (unit.position.y - enemy.position.y)**2)



def closest_enemy(enemies, unit):
    closest_enemy = enemies[0]
    closest_distance = get_distance_to(unit,closest_enemy)
    for enemy in enemies:
        distance = get_distance_to(unit, enemy)
        if distance < closest_distance:
            closest_distance = distance
            closest_enemy = enemy
    return closest_enemy