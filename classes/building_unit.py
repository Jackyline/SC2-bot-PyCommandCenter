from library import *

class BuildingUnit:
    """
    Is one building.
    Use methods defined here over methods/variables of Unit
    """

    def __init__(self, unit: Unit):
        self.unit = unit
        self.task = None
        self.has_techlab = False

    def get_unit(self):
        return self.unit

    def get_unit_type_id(self):
        return self.get_unit_type().unit_typeid

    def get_unit_type(self):
        return self.unit.unit_type

    def get_hit_points(self):
        return self.unit.hit_points

    def is_alive(self):
        return self.unit.is_alive

    def is_training(self):
        return self.unit.is_alive

    def get_tile_pos(self):
        return self.unit.tile_position

    def get_pos(self):
        return self.unit.position

    def set_task(self, task):
        self.task = task

    def get_task(self):
        return self.task

    def train(self, unit_type):
        self.get_unit().train(unit_type)