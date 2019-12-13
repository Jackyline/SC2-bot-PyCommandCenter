from classes.hmm import HiddenMarkovModel
import strategy.strategy
from library import *
from classes.task import Task
from classes.task_type import TaskType
import math

Point2D.distance = lambda self, other: math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

class ScoutingManager:

    def __init__(self, bot: IDABot):
        self.bot = bot
        self.log = {}
        self.width_ratio = 0  # Is set in on_step when map is loaded
        self.height_ratio = 0  # Is set in on_step when map is loaded
        self.columns = 0
        self.rows = 0
        self.frame_stamps = []
        self.visited = []
        self.goals = []
        self.hmm = None  # Need map size, which means it has to be created in the on_step.
        self.scouts_requested = 0
        self.enemy_base = None

        self.last_run = 0

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
                              UnitType(UNIT_TYPEID.NEUTRAL_MINERALFIELD, bot),
                              UnitType(UNIT_TYPEID.NEUTRAL_MINERALFIELD750, bot),
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
                              UnitType(UNIT_TYPEID.NEUTRAL_XELNAGATOWER, bot),
                              UnitType(UNIT_TYPEID.TERRAN_ARMORY, bot)
                              ]

    def on_step(self):


        curr_seconds = self.bot.current_frame // 10

        # Only run every 1 seconds
        if curr_seconds - self.last_run < 1:
            return

        self.last_run = curr_seconds

        # Width and height needs to be done in step because of IDABot loads map slowly.
        enemy_units = list(set(self.bot.get_all_units()) - set(self.bot.get_my_units()))
        map_width = self.bot.map_tools.width
        map_height = self.bot.map_tools.height
        scv_sight_range = UnitType(UNIT_TYPEID.TERRAN_SCV, self.bot).sight_range
        self.columns = int(math.floor(float(map_width) / scv_sight_range))
        self.rows = int(math.floor(float(map_height) / scv_sight_range))
        self.width_ratio = int(math.floor(float(map_width) / self.columns))
        self.height_ratio = int(math.floor(float(map_height) / self.rows))

        # If nr of scouts is less than 2, ask for more.

        for i in range(1 - len(self.bot.unit_manager.scout_units)):
            if self.scouts_requested < 1 and len(self.bot.unit_manager.scout_units) < 1:
                self.ask_for_scout()
                self.scouts_requested += 1

        if self.hmm is None:
            self.enemy_base = self.bot.base_location_manager.get_player_starting_base_location(
                player_constant=PLAYER_ENEMY)
            self.hmm = HiddenMarkovModel(self.columns, self.rows, self.bot.current_frame, map_height * map_width)

        if self.bot.current_frame % 250 == 0:
            self.check_for_units(enemy_units)
            self.hmm.on_step(self.log, self.bot.current_frame)
            # Reset the log so it only contains the last spotted frame,
            # no need to save what has been seen. It is already in the HMM.
            last_captured_frame = str(list(self.log.keys())[-1])
            last_captured = self.log[int(last_captured_frame)]
            self.log.clear()
            self.log[int(last_captured_frame)] = last_captured

        for scout in self.bot.unit_manager.scout_units:

            # Nothing has been spotted
            if self.hmm.get_most_likely()[0] == 0.0:
                if self.bot.unit_manager.scout_units[0].goal is None or self.enemy_base.position not in self.goals:
                    self.send_away_one_scout_to_enemy()
            else:
                if scout.reached_goal(self.bot.current_frame) or scout.get_unit().is_idle:
                    if scout.attack:
                        scout.attack = False
                        return
                    self.go_to_most_interested(scout)

    def send_away_one_scout_to_enemy(self):
        """
        In the beginning of the game, send one of the scouts to the enemy camp
        """
        self.bot.unit_manager.scout_units[0].set_goal(self.enemy_base.position)

    def ask_for_scout(self):
        task_scout = Task(TaskType.SCOUT,
                          pos=self.bot.base_location_manager.get_player_starting_base_location(PLAYER_SELF).position)
        self.bot.assignment_manager.add_task(task_scout)

    def create_log(self):
        log = []
        # Height
        for i in range(self.rows):
            # Width
            log.append([])
            for j in range(self.columns):
                log[i].append([])
        return log

    def check_for_units(self, enemy_units):
        """
        Scans the map while the scouts is out exploring
        :param enemy_units: The units spotted which belong to the enemy
        """
        time_spotted = self.bot.current_frame
        if time_spotted not in self.log:
            self.log[time_spotted] = {}
        for unit in enemy_units:
            if unit.player == PLAYER_ENEMY and \
                    unit.unit_type not in self.neutral_units:
                self.append_unit(unit, time_spotted)

    def append_unit(self, unit, time_spotted):
        """
        The map is divided into cells. By looking at the unit position, append it to the correct cell
        :param unit: A unit which belongs to enemy
        :param time_spotted: Current frame when the enemy was spotted
        """
        x = unit.tile_position.x
        y = unit.tile_position.y
        x_ratio = math.floor(x / self.width_ratio)
        y_ratio = self.rows - math.floor(y / self.height_ratio)

        tile_position = str(x_ratio) + str(y_ratio)

        if tile_position not in self.log[time_spotted]:
            self.log[time_spotted][tile_position] = []
        self.log[time_spotted][tile_position].append(unit.id)

    def go_to_most_interested(self, scout):
        """
        The HMM gives the most interesting point (Point with highetst probabilty of finding units).
        The scout gets the point as a goal to travel to
        :param scout: Scout unit.
        """
        most_likely = self.hmm.get_most_likely()
        points = most_likely[1]
        # Convert most likely cells to coordinates
        for i in range(len(points)):
            points[i] = Point2D((points[i][0] + 0.5) * self.width_ratio, ((self.rows - points[i][1]) + 0.5)
                                * self.height_ratio)
        # Send to scout, check if been visited before of the scout
        scout.check_if_visited(points, self.bot.current_frame, self.width_ratio, self.height_ratio)

    def get_enemy_target(self):
        """
        Gives the best location to attack the enemy
        :return: Point2D, attack location
        """
        most_likely = self.hmm.get_most_likely()

        # Worst case, HMM has highest prob 0.0 and scout has not reached their base for new info
        if most_likely[0] is 0.0:
            return self.enemy_base.position
        else:
            points = most_likely[1]
            for i in range(len(points)):
                points[i] = Point2D((points[i][0] + 0.5) * self.width_ratio, ((self.rows - points[i][1]) + 0.5)
                                    * self.height_ratio)
            points[0] = self.get_nearby_enemy(points[0])
            return points[0]

    def get_nearby_enemy(self, point):

        best_goal = self.enemy_base.position
        low_dist = best_goal.distance(point)
        enemy_units = list(set(self.bot.get_all_units()) - set(self.bot.get_my_units()))
        for unit in enemy_units:
            if unit.player == PLAYER_ENEMY and \
                    unit.unit_type not in self.neutral_units:
                if unit.position.distance(point) < low_dist:
                    best_goal = unit.position
                    low_dist = unit.position.distance(point)
        return best_goal


    def print_debug(self):
        last_captured_frame = '0'
        if len(self.log.keys()) > 0:
            last_captured_frame = str(list(self.log.keys())[-1])
        output = "Last scouted frame: " + last_captured_frame + '\n'
        output += "Units seen in cells: " + '\n'

        if len(self.log.keys()) > 0:
            for position, units in self.log[list(self.log.keys())[-1]].items():
                output += "[" + position[:len(position) // 2] + "]" + "[" + position[len(position) // 2:] + "]" \
                          + " = " + str(len(units))
                output += '\n'
        return output

    def print_scout_backpack(self):
        """
        Prints the backpack on the scout, containing visited point, timestamps and current goal
        """
        for scout in self.bot.unit_manager.scout_units:
            backpack = ""
            backpack += "goal: " + str(scout.get_goal()) + "\n"
            backpack += "Visited: " + str(scout.visited) + "\n"
            backpack += "Timestamps:  " + str(scout.frame_stamps)
            self.bot.map_tools.draw_text(position=scout.get_unit().position, text=backpack)

    def print_debug_prob(self):
        """
        Prints the probability in each cell on the map, where the probability is
        the likelihood of seeing an enemy unit
        """
        trans_matrix = self.hmm.get_trans_matrix()
        for i in range(0, self.columns):
            for j in range(0, self.rows):
                self.bot.map_tools.draw_text(position=Point2D((i + 0.5) * self.width_ratio, (j + 0.5) *
                                                              self.height_ratio),
                                             text="[" + str(i) + "]" + "[" + str(self.rows - j - 1) + "] = "
                                                  + str(trans_matrix[self.rows - j - 1][i]))
