from library import Point2D, TERRAN_MARINE, TERRAN_MARAUDER, TERRAN_HELLION, TERRAN_SIEGETANK, TERRAN_LIBERATOR
from  classes.state_and_reward import get_state, get_reward
from classes.q_table import QTable
import random
import copy
class MilitaryUnit:
    def __init__(self, military_unit, idabot, q_table):
        self.q_table : QTable = q_table
        self.in_combat = False
        self.unit = military_unit
        self.target = None
        self.e_in_sight = [] #enemy units in sight
        self.a_in_sight = [] #allies in sight
        self.e_in_range =  [] # enemies in range
        self.idabot = idabot
        self.state = ""
        self.old_action = 0
        self.attacking = None
        self.hp = {}  # {unit: hp}
        self.action_end_frame = idabot.current_frame

        self.learning_rate = 0.1
        self.discount_factor = 0.95 #TODO: ändra
        self.exploration = 1.0
        type_id = self.get_unit_type_id()
        if type_id == TERRAN_MARINE:
            self.attack_range = 5
            self.movement_speed = 3.15
            self.attack_frames = 5 #TODO: dubbelkolla
        elif type_id == TERRAN_MARAUDER:
            self.attack_range = 6
            self.movement_speed = 3.15
            self.attack_frames = 5
        elif type_id == TERRAN_HELLION:
            self.attack_range = 5
            self.movement_speed = 5.95
            self.attack_frames = 7
        elif type_id == TERRAN_SIEGETANK:
            self.attack_range = 7
            self.movement_speed = 3.15
            self.attack_frames = 7
        elif type_id ==TERRAN_LIBERATOR:
            self.attack_range = 10
            self.movement_speed = 4.72
            self.attack_frames = 10

            #self.temp = 0




    def on_step(self, enemies_in_sight, allies_in_sight, enemies_in_range):

        if self.state == "":
            self.update_in_sight(enemies_in_sight, allies_in_sight, enemies_in_range)
        if self.in_combat and self.idabot.current_frame >= self.action_end_frame:
            self.in_combat_on_step(enemies_in_sight, allies_in_sight, enemies_in_range)
        else:
            self.update_in_sight(enemies_in_sight, allies_in_sight, enemies_in_range)


    def in_combat_on_step(self, enemies_in_sight, allies_in_sight, enemies_in_range):

        reward = get_reward(self.hp)
        self.update_in_sight(enemies_in_sight, allies_in_sight, enemies_in_range)
        if not self.in_combat:
            return
        new_state = self.update_state()

        action_to_take = self.q_table.get_action(new_state)
        q_value = (1-self.learning_rate)*self.q_table.get_value(self.state, self.old_action) + \
                  self.learning_rate * (reward + self.discount_factor*self.q_table.get_value(new_state, action_to_take))
        self.q_table.set_value(self.state, self.old_action, q_value)

        if random.uniform(0, 1) < self.exploration:
            action_to_take = random.randint(0, 1)
        """
        self.temp += 1
        if self.temp % 6 != 0:
            self.retreat_action()
            self.action_end_frame = self.idabot.current_frame + 5
        else:
            self.attack_action()
            self.action_end_frame = self.idabot.current_frame + 5
        """

        if action_to_take == 0:
            self.retreat_action()
            self.action_end_frame = self.idabot.current_frame + 5
        elif action_to_take == 1:
            self.attack_action()
            self.action_end_frame = self.idabot.current_frame

        self.state = new_state
        self.old_action = action_to_take

    def update_in_sight(self, enemies_in_sight, allies_in_sight, enemies_in_range):
        self.e_in_sight = enemies_in_sight
        self.a_in_sight = allies_in_sight
        self.e_in_range = enemies_in_range
        #If unit was in combat but no longer is
        if len(self.e_in_sight) == 0 and self.in_combat == True:
            self.action_end_frame = self.idabot.current_frame
            self.stop()

        self.in_combat = True if enemies_in_sight else False

        for unit in self.e_in_sight:
            self.hp[unit] = unit.hit_points
        for unit in self.a_in_sight:
            self.hp[unit.get_unit()] = unit.get_unit().hit_points


    def update_state(self):
        health = self.get_health()
        on_cooldown = self.get_weapon_on_cooldown()
        distance_to_closest_enemy = self.get_distance_to(self.get_closest_enemy())
        enemies = len(self.e_in_sight)
        allies = len(self.a_in_sight) + 1
        return get_state(health, on_cooldown, distance_to_closest_enemy, enemies, allies)


    def attack_action(self):
        self.attacking = self.lowest_health_enemy()
        self.attack_unit(self.attacking)


    def retreat_action(self):
        position = self.calculate_position()
        self.move_to(position)


    def calculate_position(self):
        my_pos = self.get_position()
        enemy_pos_lst = [enemy.position for enemy in self.e_in_sight]
        position = Point2D()
        position.x = my_pos.x
        position.y = my_pos.y
        for enemy_pos in enemy_pos_lst:
            position += (position + enemy_pos*(-1))* (1/(0.5+self.idabot.map_tools.get_ground_distance(position, enemy_pos)))
        while not self.idabot.map_tools.is_walkable(round(position.x), round(position.y)) and position.x < self.idabot.map_tools.width and \
                position.y < self.idabot.map_tools.height:

            position = position * 1.1
        # TODO hur ökar man längden på vektorn????
        print("len(position): ", self.idabot.map_tools.get_ground_distance(my_pos, position) )
        print("my position: ", my_pos, "position: ", position,"position *2: ", position * 1.2)
        return position

    def move_to(self, location):
        self.get_unit().move(location)

    def lowest_health_enemy(self):
        if len(self.e_in_range) > 0:
            lowest_health = self.e_in_range[0]
            for enemy in self.e_in_range:
                if enemy.hit_points < lowest_health.hit_points:
                    lowest_health = enemy
            return lowest_health
        else:
            lowest_health = self.e_in_sight[0]
            for enemy in self.e_in_sight:
                if enemy.hit_points < lowest_health.hit_points:
                    lowest_health = enemy
            return lowest_health


    def stop(self):
        self.get_unit().stop()

    def attack_unit(self, enemy):
        self.get_unit().attack_unit(enemy)

    def get_closest_enemy(self):
        if len(self.e_in_sight) == 0:
            return None
        closest_enemy = self.e_in_sight[0]
        closest_distance = self.get_distance_to(closest_enemy)
        for enemy in self.e_in_sight:
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
