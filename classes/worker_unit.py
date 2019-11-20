class WorkerUnit:
    def __init__(self, worker_unit):
        self.unit = worker_unit
        self.current_work_station = None
        self.current_work_position = None
        self.building = False

    def set_idle(self):
        self.current_work_station = None
        self.current_work_position = None
        self.building = False

        self.unit.hold_position()

    def get_id(self):
        return self.unit.id

    def get_unit_type(self):
        return self.unit.unit_type

    def get_unit_type_id(self):
        return self.unit.unit_type.unit_typeid

    def get_job(self):
        return self.current_work_station

    def is_alive(self):
        return self.unit.is_alive

    def is_free(self):
        if not self.current_work_station and not self.building:
            return True
        else:
            return False

    def set_mining(self, mineral_field):
        self.current_work_station = mineral_field
        self.current_work_position = mineral_field.position
        self.building = False

        self.unit.right_click(mineral_field)

    def set_gassing(self, refinery):
        self.current_work_station = refinery
        self.current_work_position = refinery.position
        self.building = False

        self.unit.right_click(refinery)

    def build(self, unit_type_id, location):
        self.current_work_position = unit_type_id
        self.current_work_station = None
        self.building = True

        self.unit.build(building_type=unit_type_id, position=location)
