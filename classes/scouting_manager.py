from library import *
import numpy
import time
import math

rows = 6
columns = 7


class ScoutingManger:

    def __init__(self, bot: IDABot):
        self.bot = IDABot
        self.log = self.create_log()
        self.timestamp = 0
        self.width_ratio = 0
        self.height_ratio = 0

    def on_step(self, all_units, width, height):
        self.width_ratio = int(math.floor(float(width) / rows))
        self.height_ratio = int(math.floor(float(height) / columns))
        if time.clock() % 10000:
            print("updating")
            self.update_log(all_units)

    def update_log(self, all_units):
        for cell in self.log:
            self.check_for_units(all_units)

    def create_log(self):
        log = []
        # Height
        for i in range(0, rows, 1):
            # Width
            log.append([])
            for j in range(0, columns, 1):
                log[i].append([])
        return log

    def check_for_units(self, all_units):
        self.timestamp = time.clock()
        for unit in all_units:
            self.append_unit(unit)

    def append_unit(self, unit):
        x = unit.tile_position.x
        y = unit.tile_position.y
        x_ratio = math.floor(x / self.width_ratio)
        y_ratio = math.floor(y / self.height_ratio)
        if unit not in self.log[x_ratio][y_ratio]:
            self.log[x_ratio][y_ratio].append(unit)

    def print_debug(self):
        output = ""
        for i in range(0, rows, 1):
            count = 0
            for j in range(0, columns, 1):
                if len(self.log[i][j]) > 0:
                    count += 1
            if count > 0:
                output += "[" + str(i) + "]" + "[" + str(j) + "]" + " = " + str(count)
                output += '\n'
        print(output)
