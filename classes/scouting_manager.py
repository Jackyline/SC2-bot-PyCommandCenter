from classes.hmm import HiddenMarkovModel
from classes.scout_unit import ScoutUnit
from library import *
import numpy
import time
import math

rows = 6
columns = 6


class ScoutingManager:

    def __init__(self, bot: IDABot):
        self.bot = bot
        self.scouts = []
        self.log = {}
        self.width_ratio = 0  # Is set in on_step when map is loaded
        self.height_ratio = 0  # Is set in on_step when map is loaded
        self.hmm = None  # Need map size, which means it has to be created in the on_step.
                            
        self.neutral_units = [UnitType(UNIT_TYPEID.NEUTRAL_BATTLESTATIONMINERALFIELD, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_BATTLESTATIONMINERALFIELD750, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_COLLAPSIBLEROCKTOWERDEBRIS, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_COLLAPSIBLEROCKTOWERDIAGONAL, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_COLLAPSIBLEROCKTOWERPUSHUNIT, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_COLLAPSIBLETERRANTOWERDEBRIS, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_COLLAPSIBLETERRANTOWERDIAGONAL, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_COLLAPSIBLETERRANTOWERPUSHUNIT, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_COLLAPSIBLETERRANTOWERPUSHUNITRAMPLEFT, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_COLLAPSIBLETERRANTOWERPUSHUNITRAMPRIGHT, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_COLLAPSIBLETERRANTOWERRAMPLEFT, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_COLLAPSIBLETERRANTOWERRAMPRIGHT, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_DEBRISRAMPLEFT, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_DEBRISRAMPRIGHT, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_DESTRUCTIBLEDEBRIS6X6, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_DESTRUCTIBLEDEBRISRAMPDIAGONALHUGEBLUR, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_DESTRUCTIBLEDEBRISRAMPDIAGONALHUGEULBR, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_DESTRUCTIBLEROCK6X6, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_FORCEFIELD, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_KARAKFEMALE, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_LABMINERALFIELD, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_LABMINERALFIELD750, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_MINERALFIELD, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_MINERALFIELD750, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_PROTOSSVESPENEGEYSER, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_PURIFIERMINERALFIELD, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_PURIFIERMINERALFIELD750, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_PURIFIERRICHMINERALFIELD, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_PURIFIERRICHMINERALFIELD750, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_PURIFIERVESPENEGEYSER, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_RICHMINERALFIELD750, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_RICHVESPENEGEYSER, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_SCANTIPEDE, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_SHAKURASVESPENEGEYSER, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_SPACEPLATFORMGEYSER, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_UNBUILDABLEBRICKSDESTRUCTIBLE, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_UNBUILDABLEPLATESDESTRUCTIBLE, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_UTILITYBOT, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_VESPENEGEYSER, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_XELNAGATOWER, bot)
                              ]

    def on_step(self, available_scouts):
        # Width and height needs to be done in step because of IDABot loads map slowly.
        enemy_units = list(set(self.bot.get_all_units()) - set(self.bot.get_my_units()))
        map_width = self.bot.map_tools.width
        map_height = self.bot.map_tools.height
        self.width_ratio = int(math.floor(float(map_width) / rows))
        self.height_ratio = int(math.floor(float(map_height) / columns))

        # If nr of scouts is less than 2, ask for more.
        # This will be different in the end, CHANGE THIS LATER!

        enemy_base = self.bot.base_location_manager.get_player_starting_base_location(
            player_constant=PLAYER_ENEMY)


        if len(self.scouts) <= 2:
            self.ask_for_scout(available_scouts)

        if self.hmm is None:
            self.hmm = HiddenMarkovModel(columns, rows, self.bot.current_frame, map_height * map_width)

        if self.bot.current_frame % 400 == 0:
            self.check_for_units(enemy_units)
            self.hmm.on_step(self.log, self.bot.current_frame)

        for scout in self.scouts:
            if not scout.is_alive:
                self.scouts.remove(scout)

            # Nothing has been spotted
            if self.hmm.get_most_likely()[0] == 0.0:
                if self.scouts[0].is_idle():
                    self.send_away_one_scout_to_enemy()
            else:
                if scout.is_idle() or scout.reach_goal():
                    self.go_to_most_interested(scout)

    def send_away_one_scout_to_enemy(self):
        enemy_base = self.bot.base_location_manager.get_player_starting_base_location(
            player_constant=PLAYER_ENEMY)
        self.scouts[0].set_goal(enemy_base.position, self.bot.current_frame)

    def ask_for_scout(self, available_scouts):
        for i in range(0, 2):
            self.scouts.append(ScoutUnit(available_scouts[i].unit))

    def create_log(self):
        log = []
        # Height
        for i in range(rows):
            # Width
            log.append([])
            for j in range(columns):
                log[i].append([])
        return log

    def check_for_units(self, all_units):
        time_spotted = self.bot.current_frame
        if time_spotted not in self.log:
            self.log[time_spotted] = {}
        for unit in all_units:
            if unit.player == PLAYER_ENEMY and unit.unit_type not in self.neutral_units:
                self.append_unit(unit, time_spotted)

    def append_unit(self, unit, time_spotted):
        x = unit.tile_position.x
        y = unit.tile_position.y
        x_ratio = math.floor(x / self.width_ratio)
        y_ratio = math.floor(y / self.height_ratio)

        print("X AND Y IS: " + str(x) + "  " + str(y) + "   TRANSFORM BACK TO:  " + str(x_ratio*self.width_ratio) + " " + str(y_ratio*self.height_ratio))
        tile_position = str(x_ratio) + str(y_ratio)

        if tile_position not in self.log[time_spotted]:
            self.log[time_spotted][tile_position] = []
        self.log[time_spotted][tile_position].append(unit.id)

    def go_to_most_interested(self, scout):
        most_likely = self.hmm.get_most_likely()
        points = most_likely[1]
        for i in range(len(points)):
            print("MOST INTR POINT IS:    " + str(points[i][0] * self.width_ratio) + "  " + str((columns - points[i][1]) * self.height_ratio))
            points[i] = Point2DI(points[i][0] * self.width_ratio, (columns - points[i][1]) * self.height_ratio)
        scout.check_if_visited(points, self.bot.current_frame)

    def print_debug(self):
        last_captured_frame = '0'
        if len(self.log.keys()) > 0:
            last_captured_frame = str(list(self.log.keys())[-1])
        output = "Last scouted frame: " + last_captured_frame + '\n'
        output += "Units seen in cells: " + '\n'

        if len(self.log.keys()) > 0:
            for position, units in self.log[list(self.log.keys())[-1]].items():
                output += "[" + position[0] + "]" + "[" + position[1] + "]" + " = " + str(len(units))
                output += '\n'
        return output
