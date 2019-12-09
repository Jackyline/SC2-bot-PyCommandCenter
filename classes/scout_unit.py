from library import *
import random
import math

Point2D.distance = lambda self, other: math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
Point2D.equal = lambda self, other: self.x == other.x and self.y == other.y


class ScoutUnit:
    def __init__(self, scout_unit, scout_manager, strategy_manager, number):
        self.unit = scout_unit
        self.goal = None
        self.manager = scout_manager
        self.strategy_manager = strategy_manager
        self.num = number

    def get_unit(self):
        return self.unit

    def get_goal(self):
        return self.goal

    def get_visited(self):
        return self.manager.visited

    def get_frame_stamps(self):
        return self.manager.frame_stamps

    def is_idle(self):
        return self.unit.is_idle

    def is_alive(self):
        return self.unit.is_alive

    def reached_goal(self, current_frame):
        if self.goal is None:
            return True
        else:
            if self.unit.position.distance(self.goal) < 5:
                self.manager.visited.append(self.goal)
                self.manager.frame_stamps.append(current_frame)
                return True
            else:
                return False

    def set_goal(self, goal):
        self.unit.move(goal)
        self.goal = goal

    def check_if_visited(self, goals, current_frame, width_ratio, height_ratio, columns):
        for point in goals:
            if not self.check_in_visited(point):
                self.set_goal(point)
            else:
                # Check how long time it was since first discovery and go there if it is been more than 2000 frames
                # since last time
                first_time_visited = self.manager.frame_stamps[0]
                if (current_frame - first_time_visited) > 4000:
                    first_visited = self.manager.visited[0]
                    self.check_in_visited(point, True)
                    self.set_goal(first_visited)
                else:
                    # If we just spotted our first discover recently, go random.
                    goal = self.set_goal_strategy(width_ratio, height_ratio)
                    self.set_goal(goal)

    def check_in_visited(self, point, *args):
        for visited in self.manager.visited:
            if visited.equal(point):
                # args is used to remove object, called from set_goal
                if args:
                    index = self.manager.visited.index(visited)
                    self.manager.visited.remove(visited)
                    self.manager.frame_stamps.pop(index)
                return True
        return False

    def set_goal_strategy(self, width_ratio, height_ratio):
        """
        Sets random explore point as goal depending on our strategy and base location
        :param width_ratio: width ratio on map, given from scout manager
        :param height_ratio: height ratio on map, given from scout manager
        """
        y_base_cell = math.floor(self.manager.bot.base_location_manager.get_player_starting_base_location(
            player_constant=PLAYER_SELF).position.y / height_ratio)
        strategy = self.strategy_manager.get_strategy()
        goal = None
        if strategy == "Defensive":
            # Top corner base
            if y_base_cell - 16 > 0 and self.num is 1:
                pos = self.rand_loc((2, 2), (16, 19), (2, 19))
                goal = Point2D((pos[0] + 0.5) * width_ratio, (pos[1] + 0.5) * height_ratio)
            # Low corner base
            else:
                pos = self.rand_loc((2, 2), (16, 19), (16, 2))
                goal = Point2D((pos[0] + 0.5) * width_ratio, (pos[1] + 0.5) * height_ratio)

        elif strategy == "Offensive":
            # Top corner base
            if y_base_cell - 16 > 0 and self.num is 1:
                pos = self.rand_loc((2, 2), (16, 19), (16, 2))
                goal = Point2D((pos[0] + 0.5) * width_ratio, (pos[1] + 0.5) * height_ratio)
            # Low corner base
            else:
                pos = self.rand_loc((2, 2), (16, 19), (2, 19))
                goal = Point2D((pos[0] + 0.5) * width_ratio, (pos[1] + 0.5) * height_ratio)

        return goal

    def rand_loc(self, pt1, pt2, pt3):
        """
        Random point within the base location
        """
        b, t = sorted([random.random(), random.random()])
        return (math.floor(b * pt1[0] + (t - b) * pt2[0] + (1 - t) * pt3[0]),
                math.floor(b * pt1[1] + (t - b) * pt2[1] + (1 - t) * pt3[1]))
