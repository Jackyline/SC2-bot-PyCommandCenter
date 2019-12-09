
import math
from library import PLAYER_ENEMY, PLAYER_SELF

def get_state(health, on_cooldown : bool, distance_to_closest_enemy, enemies, allies):
    """
    Gets the state based on the parameters, change the numbers in the return statemets to decide intervalls
    :param health: percentage of health of the unit 0 < health <= 1
    :param on_cooldown: is weapon on cooldown?
    :param distance_to_closest_enemy: distance to the closest enemy (in sight), int between 0 and 10
    :param enemies: number of enemies in sight
    :param allies: number of allies in sight
    :return: the state (str)
    """
    def calc_health(intervall_nr) -> str:
        intervall = 1/intervall_nr
        return str(math.floor(health/intervall) if health < 1 else str(intervall_nr-1))

    def calc_on_cooldown() -> str:
        return "0" if on_cooldown else "1"

    def calc_distance_to_closest_enemy(intervall_nr) -> str:
        return str(distance_to_closest_enemy)

    def calc_enemy_ally_ratio(intervall_nr) -> str:
        intervall = 100 / intervall_nr
        return str(math.floor((100 * enemies / (allies + enemies)) / intervall))

    return "" + calc_health(4) + calc_on_cooldown() + calc_distance_to_closest_enemy(4) + calc_enemy_ally_ratio(4)

def get_state_marine(health, on_cooldown, distance_to_closest_enemy, e_that_can_attack, allies, enemies):
    """
    Gets the state based on the parameters, change the numbers in the return statemets to decide intervalls
    :param health: percentage of health of the unit 0 < health <= 1
    :param on_cooldown: is weapon on cooldown?
    :param distance_to_closest_enemy: distance to the closest enemy (in sight), int between 0 and 10
    :param enemies: number of enemies in sight
    :param allies: number of allies in sight
    :return: the state (str)
    """
    def calc_health(intervall_nr) -> str:
        intervall = 1/intervall_nr
        return str(math.floor(health/intervall) if health < 1 else str(intervall_nr-1))

    def calc_on_cooldown() -> str:
        return str(1 if on_cooldown else 0)

    def calc_distance_to_closest_enemy() -> str:
        return str(distance_to_closest_enemy) if distance_to_closest_enemy < 11 else "11"

    def calc_e_that_can_attack() -> str:
        return str(e_that_can_attack)

    def calc_enemy_ally_ratio(intervall_nr) -> str:
        intervall = 100 / intervall_nr
        return str(math.floor((100 * enemies / (allies + enemies)) / intervall))



    state = "" + \
            calc_health(4) + \
            calc_on_cooldown() + \
            calc_distance_to_closest_enemy() + \
            calc_e_that_can_attack() + \
            calc_enemy_ally_ratio(5)
    #print("STATE:", state)
    return state

def get_state_marauder(health, on_cooldown, distance_to_closest_enemy, e_that_can_attack, allies, enemies, concussive_shells):
    """
    Gets the state based on the parameters, change the numbers in the return statemets to decide intervalls
    :param health: percentage of health of the unit 0 < health <= 1
    :param on_cooldown: is weapon on cooldown?
    :param distance_to_closest_enemy: distance to the closest enemy (in sight), int between 0 and 10
    :param enemies: number of enemies in sight
    :param allies: number of allies in sight
    :return: the state (str)
    """
    def calc_health(intervall_nr) -> str:
        intervall = 1/intervall_nr
        return str(math.floor(health/intervall) if health < 1 else str(intervall_nr-1))

    def calc_on_cooldown() -> str:
        return str(1 if on_cooldown else 0)

    def calc_distance_to_closest_enemy() -> str:
        return str(distance_to_closest_enemy) if distance_to_closest_enemy < 11 else "11"

    def calc_e_that_can_attack() -> str:
        return str(e_that_can_attack)

    def calc_enemy_ally_ratio(intervall_nr) -> str:
        intervall = 100 / intervall_nr
        return str(math.floor((100 * enemies / (allies + enemies)) / intervall))

    def calc_concussive_shells() -> str:
        return str(1 if concussive_shells else 0)

    state = "" + \
            calc_health(4) + \
            calc_on_cooldown() + \
            calc_distance_to_closest_enemy() + \
            calc_e_that_can_attack() + \
            calc_enemy_ally_ratio(8) + \
            calc_concussive_shells()
    #print("STATE:", state)
    return state

def get_state_hellion(health, on_cooldown, distance_to_closest_enemy, e_that_can_attack, allies, enemies):
    """
    Gets the state based on the parameters, change the numbers in the return statemets to decide intervalls
    :param health: percentage of health of the unit 0 < health <= 1
    :param on_cooldown: is weapon on cooldown?
    :param distance_to_closest_enemy: distance to the closest enemy (in sight), int between 0 and 10
    :param enemies: number of enemies in sight
    :param allies: number of allies in sight
    :return: the state (str)
    """
    def calc_health(intervall_nr) -> str:
        intervall = 1/intervall_nr
        return str(math.floor(health/intervall) if health < 1 else str(intervall_nr-1))

    def calc_on_cooldown() -> str:
        return str(1 if on_cooldown else 0)

    def calc_distance_to_closest_enemy() -> str:
        return str(distance_to_closest_enemy) if distance_to_closest_enemy < 11 else "11"

    def calc_e_that_can_attack() -> str:
        return str(e_that_can_attack)

    def calc_enemy_ally_ratio(intervall_nr) -> str:
        intervall = 100 / intervall_nr
        return str(math.floor((100 * enemies / (allies + enemies)) / intervall))



    state = "" + \
            calc_health(10) + \
            calc_on_cooldown() + \
            calc_distance_to_closest_enemy() + \
            calc_e_that_can_attack() + \
            calc_enemy_ally_ratio(10)
    #print("STATE:", state)
    return state

def get_state_cyclone(health, on_cooldown, distance_to_closest_enemy, e_that_can_attack, allies, enemies):
    """
    Gets the state based on the parameters, change the numbers in the return statemets to decide intervalls
    :param health: percentage of health of the unit 0 < health <= 1
    :param on_cooldown: is weapon on cooldown?
    :param distance_to_closest_enemy: distance to the closest enemy (in sight), int between 0 and 10
    :param enemies: number of enemies in sight
    :param allies: number of allies in sight
    :return: the state (str)
    """
    def calc_health(intervall_nr) -> str:
        intervall = 1/intervall_nr
        return str(math.floor(health/intervall) if health < 1 else str(intervall_nr-1))

    def calc_on_cooldown() -> str:
        return str(1 if on_cooldown else 0)

    def calc_distance_to_closest_enemy() -> str:
        return str(distance_to_closest_enemy) if distance_to_closest_enemy < 11 else "11"

    def calc_e_that_can_attack() -> str:
        return str(e_that_can_attack)

    def calc_enemy_ally_ratio(intervall_nr) -> str:
        intervall = 100 / intervall_nr
        return str(math.floor((100 * enemies / (allies + enemies)) / intervall))



    state = "" + \
            calc_health(8) + \
            calc_on_cooldown() + \
            calc_distance_to_closest_enemy() + \
            calc_e_that_can_attack() + \
            calc_enemy_ally_ratio(8)
    #print("STATE:", state)
    return state

def get_reward(units_hp: dict):
    """

    :param units_hp: {unit, old_hp}
    :return: reward
    """
    reward = 0
    for unit, old_hp in units_hp.items():
        if unit.player == PLAYER_SELF:
            reward -= (old_hp - unit.hit_points)
            if not unit.is_alive:
                reward -= 10
        else:
            reward += (old_hp - unit.hit_points)
            if not unit.is_alive:
                reward += 10
    #print("REWARD", reward)
    return reward
