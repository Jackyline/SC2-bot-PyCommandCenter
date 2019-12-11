from library import *
from  classes.state_and_reward import get_state, get_reward, get_state_marauder, get_state_marine, get_state_hellion, \
    get_state_cyclone
from classes.q_table import QTable
import random
import copy
import math
class MilitaryUnit:
    """
    Class for handling a military unit.

    There is a problem with distance where the value from get_ground_distance does not correlate with the units for
    attack_range or sight_range. sight_range and attack_range are longer in distance units than specified in the api
    """
    def __init__(self, military_unit, idabot, q_table):
        self.q_table : QTable = q_table  # The Q-table for this specific type of unit
        self.unit = military_unit
        self.idabot = idabot

        self.e_in_sight = []
        self.e_that_can_attack = []  # enemy units in sight
        self.a_in_sight = []  # allies in sight
        self.e_in_range = []  # enemies in range

        self.in_combat = False # A unit is in combat if it is within 10 distance of any enemy units
        self.first_tick_in_combat = False
        self.target = None
        self.attacked = False # was the last action an attack?
        self.hp = {}  # {unit: hp}  # used to compare hp of units between two ticks

        # Q-learning variables
        self.old_action = 0  # The latest action taken by this unit
        self.state = ""
        self.learning_rate = 0.1
        self.discount_factor = 0.9 #TODO: ändra
        self.exploration = 0.0 # Set this to 0 to use the learned policy
        self.total_reward = 0


        self.action_end_frame = idabot.current_frame # Used to set how long until the action should be updated
        self.concussive_shells = False  # The marauder ability "concussive shells", #TODO: se till att detta sätts när den är gäller, fråga dylan?

        self.attack_range = self.get_unit_type().attack_range
        self.movement_speed = self.get_unit_type().movement_speed
        self.sight_range = self.get_unit_type().sight_range

        type_id = self.get_unit_type_id()
        # Specifika variabler för olika enheter
        if type_id == UNIT_TYPEID.TERRAN_MARINE:
            self.attack_animation_offset = 0
        elif type_id == UNIT_TYPEID.TERRAN_MARAUDER:
            self.concussive_shells = True  # remove comment when concussive_shells are researched
            self.attack_animation_offset = 7
        elif type_id == UNIT_TYPEID.TERRAN_HELLION:
            self.attack_animation_offset = 0
        elif type_id == UNIT_TYPEID.TERRAN_CYCLONE:
            self.attack_animation_offset = 0
        elif type_id == UNIT_TYPEID.TERRAN_MEDIVAC:
            self.attack_animation_offset = -1
        elif type_id == UNIT_TYPEID.TERRAN_THOR:
            self.attack_animation_offset = -1
        elif type_id == UNIT_TYPEID.TERRAN_BATTLECRUISER:
            self.attack_animation_offset = -1
        elif type_id == UNIT_TYPEID.TERRAN_REAPER:
            self.attack_animation_offset = -1
        elif type_id == UNIT_TYPEID.TERRAN_WIDOWMINE:
            self.attack_animation_offset = -1
        elif type_id == UNIT_TYPEID.TERRAN_RAVEN:
            self.attack_animation_offset = -1
        elif type_id == UNIT_TYPEID.TERRAN_SIEGETANK:
            self.attack_animation_offset = -1
        elif type_id == UNIT_TYPEID.TERRAN_SIEGETANKSIEGED:
            self.attack_animation_offset -= 1
        elif type_id == UNIT_TYPEID.TERRAN_VIKINGFIGHTER:
            self.attack_animation_offset = -1
        elif type_id == UNIT_TYPEID.TERRAN_LIBERATOR:
            self.attack_animation_offset = -1
        elif type_id == UNIT_TYPEID.TERRAN_BANSHEE:
            self.attack_animation_offset = -1
        elif type_id == UNIT_TYPEID.TERRAN_GHOST:
            self.attack_animation_offset = -1

    def on_step(self, e_in_sight, enemies_that_can_attack, allies_in_sight, enemies_in_range):
        """
        :param enemies_in_sight: all enemy combat units within a range of 10 distance units
        :param allies_in_sight: all allies within 10 distance units
        :param enemies_in_range: all enemies within a range of 6
        :return: None
        """
        self.update_in_sight(e_in_sight, enemies_that_can_attack, allies_in_sight, enemies_in_range)
        if not self.in_combat:
            if self.attack_animation_offset == -1:
                self.not_trained_on_step()
            """
            closest_enemy = self.__get_closest_enemy(self.idabot.unit_manager.visible_enemies)
            if closest_enemy:
                self.attack_unit(closest_enemy)
            """
        elif self.attack_animation_offset == -1:
            self.in_combat_on_step_not_trained(e_in_sight, enemies_that_can_attack, allies_in_sight, enemies_in_range)

        elif self.attacked and self.get_weapon_cooldown() == 0:
            self.action_end_frame += 1

        elif self.idabot.current_frame > self.action_end_frame:
            self.in_combat_on_step(e_in_sight, enemies_that_can_attack, allies_in_sight, enemies_in_range)

        elif self.attacked and self.get_weapon_cooldown() > 0 and len(e_in_sight) > 0:
            self.retreat_action()

    def not_trained_on_step(self):
        if self.get_unit_type_id() == UNIT_TYPEID.TERRAN_SIEGETANKSIEGED:
            self.get_unit().ability(ABILITY_ID.MORPH_UNSIEGE)

    def in_combat_on_step_not_trained(self, e_in_sight, enemies_that_can_attack, allies_in_sight, enemies_in_range):
        if self.get_unit_type_id() == UNIT_TYPEID.TERRAN_SIEGETANK:
            self.get_unit().ability(ABILITY_ID.MORPH_SIEGEMODE)

    def in_combat_on_step(self, e_in_sight, enemies_that_can_attack, allies_in_sight, enemies_in_range):
        """
        The on step function for when a unit is in combat,
        (almost) everything that concerns attack pattern is handled here.
        :param enemies_in_sight:  all enemy combat units within a range of 10 distance units
        :param allies_in_sight: all allies within 10 distance units
        :param enemies_in_range:  all enemies within a range of 6 #TODO: se on_step
        :return: None
        """
        self.attacked = False
        reward = get_reward(self.hp)  # reward must be calculated before self.hp is updated (which happens in update_in_sight())
        #if self.get_weapon_cooldown() == 0: # This is to give the unit a little more incentive to attack
            #reward -= 5

        self.total_reward += reward # To keep track of the total reward for debugging putposes
        self.update_hp()

        new_state = self.update_state()  # Get new state, note that self.state still refers to the previous state
        action_to_take = self.q_table.get_action(new_state)  # Get next action to take

        if random.uniform(0, 1) < self.exploration:
            action_to_take = random.randint(0, 1)

        if self.first_tick_in_combat:
            action_to_take = 0
            self.first_tick_in_combat = False

        if self.exploration > 0: # Only if exploration is on should the Q-table be updated
            q_value = (1-self.learning_rate)*self.q_table.get_value(self.state, self.old_action) + \
                      self.learning_rate * (reward + self.discount_factor*self.q_table.get_value(new_state, action_to_take))
            self.q_table.set_value(self.state, self.old_action, q_value)



        if action_to_take == 0:
            #print("attack")
            self.attack_action()
            self.attacked = True
            if self.get_weapon_cooldown() == 0:
                self.action_end_frame = self.idabot.current_frame + self.attack_animation_offset
            else:
                self.action_end_frame = self.idabot.current_frame
        elif action_to_take == 1:
            #print("retreat")
            self.retreat_action()
            self.action_end_frame = self.idabot.current_frame

        self.state = new_state
        self.old_action = action_to_take

    def update_in_sight(self, e_in_sight, enemies_that_can_attack, allies_in_sight, enemies_in_range):
        """
        Updates if a unit is in combat or not as well as the lists with nearby units
        :param enemies_in_sight: List with enemy combat units within a 10 distance unit radius
        :param allies_in_sight: List with ally combat units within a 10 distance unit radius
        :param enemies_in_range: List with enemy combat units within a 6 distance unit radius # TODO: bör vara individuellt, se on_step
        :return: None
        """
        self.e_in_sight = e_in_sight
        self.e_that_can_attack = enemies_that_can_attack
        self.a_in_sight = allies_in_sight
        self.e_in_range = enemies_in_range

        #If unit was in combat but no longer is
        if len(self.e_in_sight) == 0 and self.in_combat == True:
            self.action_end_frame = 0
            self.stop()
        elif len(self.e_in_sight) > 0 and self.in_combat == False:
            self.action_end_frame = 0
            self.stop()
            self.first_tick_in_combat = True
        self.in_combat = True if e_in_sight else False

    def update_hp(self):
        # Updates self.hp
        # Enemy units
        self.hp.clear()
        for unit in self.e_in_sight:
            self.hp[unit] = unit.hit_points
        # Allied units
        for unit in self.a_in_sight:
            self.hp[unit.get_unit()] = unit.get_unit().hit_points
        # self
        self.hp[self.get_unit()] = self.get_unit().hit_points

    def update_state(self):
        """
        gets a new state, different state functions are called depending on the unit type
        :return: state : str
        """

        health = self.get_health()  # percentage of max_hit_points
        on_cooldown = self.get_weapon_cooldown() != 0 # cooldown time of weapon in ticks
        distance_to_closest_enemy = math.floor(self.get_distance_to(self.get_closest_enemy())) # will be between 0 and 10
        enemies = len(self.e_in_sight)
        allies = len(self.a_in_sight) + 1 # +1 for self
        enemies_that_can_attack = len(self.e_that_can_attack)
        if self.get_unit_type_id() == UNIT_TYPEID.TERRAN_MARAUDER:
            return get_state_marauder(health, on_cooldown, distance_to_closest_enemy, enemies_that_can_attack, allies, enemies, self.concussive_shells)
        elif self.get_unit_type_id() == UNIT_TYPEID.TERRAN_MARINE:
            return get_state_marine(health, on_cooldown, distance_to_closest_enemy, enemies_that_can_attack, allies, enemies)
        elif self.get_unit_type_id() == UNIT_TYPEID.TERRAN_HELLION:
             return get_state_hellion(health, on_cooldown, distance_to_closest_enemy, enemies_that_can_attack, allies, enemies)
        elif self.get_unit_type_id() == UNIT_TYPEID.TERRAN_CYCLONE:
            return get_state_cyclone(health, on_cooldown, distance_to_closest_enemy, enemies_that_can_attack, allies, enemies)

    def attack_action(self):
        """
        Attacks the enemy in range with lowest health
        :return:
        """
        self.attack_unit(self.lowest_health_enemy())

    def retreat_action(self):
        """
        Moves away from enemies
        :return:
        """
        position = self.calculate_position()
        self.move_to(position)


    def calculate_position(self):
        """
        Calculates the position to retreat to
        :return:
        """
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
        pos_diffx = (position.x - my_pos.x)
        pos_diffy = (position.y - my_pos.y)
        position.x += pos_diffx * 5
        position.y += pos_diffy * 5
        return position

    def move_to(self, location):
        self.get_unit().move(location)

    def lowest_health_enemy(self):
        """
        Gets the enemy with lowest health in range,
        if there are none in range, the closest enemy in sight will be returned
        :return: unit with lowest health
        """
        if len(self.e_in_range) > 0:
            lowest_health = self.e_in_range[0]
            for enemy in self.e_in_range:
                if enemy.hit_points < lowest_health.hit_points:
                    lowest_health = enemy
            return lowest_health
        else:
            lowest_health = self.get_closest_enemy()

            return lowest_health


    def attack_unit(self, enemy):
        self.get_unit().attack_unit(enemy)

    def attack_move(self, position: Point2D):
        self.get_unit().attack_move(position)

    def get_closest_enemy(self):
        """
        gets the closest enemy
        :return: closest unit
        """
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

    def __get_closest_enemy(self,unit_list): #TODO: OUTDATED, to be removed
        """
        gets the closest enemy
        :return: closest unit
        """
        if len(unit_list) == 0:
            return None
        closest_enemy = unit_list[0]
        closest_distance = self.get_distance_to(closest_enemy)
        for enemy in unit_list:
            distance = self.get_distance_to(enemy)
            if  distance < closest_distance:
                closest_distance = distance
                closest_enemy = enemy
        return closest_enemy

    def get_distance_to(self, unit):
        """
        Return the distance to a unit, if units is none 10 will be returned
        :param unit:
        :return:
        """
        if unit:
            return math.sqrt((self.get_position().x - unit.position.x)**2 +
                                     (self.get_position().y - unit.position.y)**2)
        return 10

    def update_unit_type(self, unit):
        self.unit = unit
        self.attack_range = self.get_unit_type().attack_range
        self.movement_speed = self.get_unit_type().movement_speed
        self.sight_range = self.get_unit_type().sight_range


    def stop(self):
        self.get_unit().stop()

    def get_position(self):
        return self.get_unit().position

    def get_id(self):
        return self.unit.id

    def is_in_combat(self):
        return self.in_combat

    def get_health(self):
        return  self.get_unit().hit_points / self.get_unit().max_hit_points

    def get_unit(self):
        return self.unit

    def get_weapon_cooldown(self):
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

    def get_task(self):
        return self.task

    def set_task(self, task):
        self.task = task
