
from  classes.state_and_reward import get_state, get_reward
from classes.q_table import QTable
import random
class MilitaryUnit:
    def __init__(self, military_unit, idabot, q_table):
        self.q_table : QTable = q_table
        self.in_combat = False
        self.unit = military_unit
        self.target = None
        self.e_in_range = [] #enemy units in range
        self.a_in_range = [] #allies in range
        self.idabot = idabot
        self.state = ""
        self.old_action = 0
        self.attacking = None
        self.hp = {}  # {unit: hp}

        self.learning_rate = 0.1
        self.discount_factor = 0.95 #TODO: ändra
        self.exploration = 0.0

    def on_step(self, enemies_in_range, allies_in_range):
        if self.state == "":
            self.update_in_range(enemies_in_range,allies_in_range)
        if self.in_combat:
            self.in_combat_on_step(enemies_in_range, allies_in_range)
        else:
            self.update_in_range(enemies_in_range, allies_in_range)


    def in_combat_on_step(self, enemies_in_range, allies_in_range):

        reward = get_reward(self.hp)
        self.update_in_range(enemies_in_range, allies_in_range)
        if not self.in_combat:
            return
        new_state = self.update_state()

        action_to_take = self.q_table.get_action(new_state)
        q_value = (1-self.learning_rate)*self.q_table.get_value(self.state, self.old_action) +\
                  self.learning_rate * (reward + self.discount_factor*self.q_table.get_value(new_state, action_to_take))
        self.q_table.set_value(self.state, self.old_action, q_value)

        if random.uniform(0, 1) < self.exploration:
            action_to_take = random.randint(0, 1)
        if action_to_take == 0:

            self.retreat_action()
        elif action_to_take == 1:

            self.attack_action()

        self.state = new_state
        self.old_action = action_to_take

    def update_in_range(self, enemies_in_range, allies_in_range):
        self.e_in_range = enemies_in_range
        self.a_in_range = allies_in_range

        #If unit was in combat but no longer is
        if len(self.e_in_range) == 0 and self.in_combat == True:
            self.stop()

        self.in_combat = True if enemies_in_range else False

        for unit in self.e_in_range:
            self.hp[unit] = unit.hit_points
        for unit in self.a_in_range:
            self.hp[unit.get_unit()] = unit.get_unit().hit_points


    def update_state(self):
        health = self.get_health()
        on_cooldown = self.get_weapon_on_cooldown()
        distance_to_closest_enemy = self.get_distance_to(self.get_closest_enemy())
        enemies = len(self.e_in_range)
        allies = len(self.a_in_range) +1
        return get_state(health, on_cooldown, distance_to_closest_enemy, enemies, allies)


    def attack_action(self):
        self.attacking = self.lowest_health_enemy()
        self.attack_unit(self.attacking)


    def retreat_action(self):
        position = self.calculate_position()
        self.move_to(position)


    def calculate_position(self):
        my_pos = self.get_position()
        enemy_pos_lst = [enemy.position for enemy in self.e_in_range]
        position = my_pos
        for enemy_pos in enemy_pos_lst:
            position += (my_pos + enemy_pos*(-1))* (1/(0.5+self.idabot.map_tools.get_ground_distance(my_pos, enemy_pos)))
        if self.idabot.map_tools.get_ground_distance(my_pos, position) != 0:
            position = position * (1 / self.idabot.map_tools.get_ground_distance(my_pos, position))
        while not self.idabot.map_tools.is_walkable(round(position.x), round(position.y)) and position.x < self.idabot.map_tools.width and \
                position.y < self.idabot.map_tools.height:
            position = position * 1.1
        return position


    def move_to(self, location):
        self.get_unit().move(location)

    def lowest_health_enemy(self):
        lowest_health = self.e_in_range[0]
        for enemy in self.e_in_range:
            if enemy.hit_points < lowest_health.hit_points:
                lowest_health = enemy
        return lowest_health




    def stop(self):
        self.get_unit().stop()


    def attack_unit(self, enemy):
        self.get_unit().attack_unit(enemy)

    def get_closest_enemy(self):
        if len(self.e_in_range) == 0:
            return None
        closest_enemy = self.e_in_range[0]
        closest_distance = self.get_distance_to(closest_enemy)
        for enemy in self.e_in_range:
            distance = self.get_distance_to(enemy)
            if  distance < closest_distance:
                closest_distance = distance
                closest_enemy = enemy
        return closest_enemy

    def get_distance_to(self, unit):
        if unit:
            return self.idabot.map_tools.get_ground_distance(self.get_position(), unit.position)
        return 10

    def get_position(self):
        return self.get_unit().position

    def get_id(self):
        return self.unit.id


    def is_in_combat(self):
        return self.in_combat

    def get_health(self):
        return self.get_unit().hit_points

    def get_unit(self):
        return self.unit

    def get_weapon_on_cooldown(self):
        return self.get_unit().weapon_cooldown

    def get_unit_type(self):
        return self.unit.unit_type

    def get_unit_type_id(self):
        return self.unit.unit_type.unit_typeid

    def get_job(self):
        return self.unit.target

    def is_alive(self):
        return self.unit.is_alive

    def is_free(self):
        if not self.target:
            return True
        else:
            return False

    def set_attack_target(self, target):
        self.target = target
        self.unit.attack_unit(target)

    def set_job(self, location):
        self.target = None
        self.unit.attack_move(location)
