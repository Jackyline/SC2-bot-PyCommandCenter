from classes.task_type import TaskType
from library import *


class Task:
    """
    TODO
    """
    def __init__(self, task_type : TaskType, pos: Unit.tile_position):
        self.task_type = task_type
        self.pos = pos

