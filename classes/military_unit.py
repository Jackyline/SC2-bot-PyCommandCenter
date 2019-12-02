
class MilitaryUnit:
    def __init__(self, military_unit):
        self.in_combat = False
        self.unit = military_unit
        self.target = None

    def update_in_combat(self, in_combat):
         self.in_combat = in_combat

    def get_id(self):
        return self.unit.id

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

    def get_task(self):
        return self.task

    def set_task(self, task):
        self.task = task
