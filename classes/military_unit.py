
class MilitaryUnit:
    def __init__(self, military_unit, idabot, q_table):
        self.q_table = q_table
        self.in_combat = False
        self.unit = military_unit
        self.target = None
        self.in_range = [] #enemy units in range
        self.idabot = idabot

    def on_step(self):
        if self.in_combat:
            self.in_combat_on_step()

    def in_combat_on_step(self):
        state = self.get_state()
        #self.attack_action()
        self.retreat_action()

    def get_state(self):
        pass

    def get_state_health(self):
        pass

    def get_state_weapon_cool_down(self):
        pass

    def get_state_enemies_in_range(self):
        pass



    def attack_action(self):
        self.attack_unit(self.lowest_health_enemy())

    def retreat_action(self):
        position = self.calculate_position()
        self.move_to(position)




    def calculate_position(self):
        my_pos = self.get_position()
        enemy_pos_lst = [enemy.position for enemy in self.in_range]
        position = my_pos
        for enemy_pos in enemy_pos_lst:
            position += (my_pos + enemy_pos*(-1))* (1/(0.5+self.idabot.map_tools.get_ground_distance(my_pos, enemy_pos)))
        if self.idabot.map_tools.get_ground_distance(my_pos, position) != 0:
            position = position * (1 / self.idabot.map_tools.get_ground_distance(my_pos, position))
        while not self.idabot.map_tools.is_walkable(round(position.x), round(position.y)) and position.x < self.idabot.map_tools.width and \
                position.y < self.idabot.map_tools.height:
            position.x += 1
        return position


    def move_to(self, location):
        self.get_unit().move(location)

    def lowest_health_enemy(self):
        lowest_health = self.in_range[0]
        for enemy in self.in_range:
            if enemy.hit_points < lowest_health.hit_points:
                lowest_health = enemy
        return lowest_health


    def update_in_range(self, units_in_range):
        self.in_range = units_in_range
        #If unit was in combat but no longer is
        if len(self.in_range) == 0 and self.in_combat == True:
            self.stop()

        self.in_combat = True if units_in_range else False


    def stop(self):
        self.get_unit().stop()


    def attack_unit(self, enemy):
        self.get_unit().attack_unit(enemy)

    def get_position(self):
        return self.get_unit().position

    def get_id(self):
        return self.unit.id


    def is_in_combat(self):
        return self.in_combat


    def get_unit(self):
        return self.unit

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
