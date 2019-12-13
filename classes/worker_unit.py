from library import *
from classes.task_type import TaskType
class WorkerUnit:
    def __init__(self, worker_unit, idabot):
        self.unit = worker_unit
        self.task = None
        self.idabot = idabot
        self.building_max_frame = 500
        self.building_start_frame = 0

    def set_idle(self):
        self.unit.stop() #TODO: varför var denna hold_position()?

    def on_step(self):
        if self.task is None:
            return
        elif self.task.task_type is TaskType.BUILD and \
                not self.task.construct_building.unit_typeid == UNIT_TYPEID.TERRAN_COMMANDCENTER and\
                self.idabot.current_frame > (self.building_start_frame + self.building_max_frame):
            self.unit.stop()

    def is_idle(self):
        return self.unit.is_idle

    def get_id(self):
        return self.unit.id

    def get_unit(self):
        return self.unit

    def get_unit_type(self):
        return self.unit.unit_type

    def get_unit_type_id(self):
        return self.unit.unit_type.unit_typeid

    def is_alive(self):
        return self.unit.is_alive

    def set_mining(self, mineral_field):
        self.unit.right_click(mineral_field)

    def set_gassing(self, refinery):
        self.unit.right_click(refinery)

    def build(self, unit_type_id, location):
        self.building_start_frame = self.idabot.current_frame
        location = Point2DI(int(location.x), int(location.y))
        self.unit.build(building_type=unit_type_id, position=location)

    def set_task(self, task):
        self.task = task

    def get_task(self):
        return self.task
