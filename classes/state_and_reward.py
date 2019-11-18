
import math
from library import PLAYER_ENEMY, PLAYER_SELF

def get_state(health, on_cooldown : bool, distance_to_closest_enemy, enemies, allies):
    """
    Gets the state based on the parameters, change the numbers in the return statemets to decide intervalls
    :param health: health of the unit
    :param on_cooldown: is weapon on cooldown?
    :param distance_to_closest_enemy: distance to the closest enemy (in sight), int between 0 and 10
    :param enemies: number of enemies in sight
    :param allies: number of allies in sight
    :return: the state (str)
    """
    def calc_health(intervall_nr) -> str:
        intervall = 100/intervall_nr
        return str(math.floor(health/intervall) if health < 100 else str(intervall_nr-1))

    def calc_on_cooldown() -> str:
        return "0" if on_cooldown else "1"

    def calc_distance_to_closest_enemy(intervall_nr) -> str:
        intervall = 10 / intervall_nr
        return str(math.floor(distance_to_closest_enemy/intervall)) if distance_to_closest_enemy < 10 else str(intervall_nr - 1) #TODO: tal mellan 0 och 10?

    def calc_enemy_ally_ratio(intervall_nr) -> str:
        intervall = 100 / intervall_nr
        return str(math.floor((100 * enemies / (allies + enemies)) / intervall))

    return "" + calc_health(4) + calc_on_cooldown() + calc_distance_to_closest_enemy(4) + calc_enemy_ally_ratio(4)




def get_reward(units_hp: dict):
    """

    :param units_hp: {unit, old_hp}
    :return: reward
    """
    reward = 0
    for unit, old_hp in units_hp.items():
        if unit.player == PLAYER_SELF:
            reward += (old_hp - unit.hit_points)
        else:
            reward -= (old_hp - unit.hit_points)
    return reward
