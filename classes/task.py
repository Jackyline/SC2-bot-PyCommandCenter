from classes.task_type import TaskType
from library import *


class Task:
    """
    TODO
    """

    def __init__(self, task_type: TaskType, pos: Unit.tile_position = None, construct_building=None, produce_unit=None):
        self.task_type = task_type
        self.pos = pos
        self.construct_building = construct_building
        self.produce_unit = produce_unit
