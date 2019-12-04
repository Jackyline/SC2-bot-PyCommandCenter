from library import *
import random
import math

Point2D.distance = lambda self, other: math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
Point2D.equal = lambda self, other: self.x == other.x and self.y == other.y


class ScoutUnit:
    def __init__(self, scout_unit, scout_manager):
        self.unit = scout_unit
        self.goal = None
        self.manager = scout_manager  # Keep track of frame_stamps and visited

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
                    goal = Point2D((random.randrange(2, 16) + 0.5) * width_ratio, (random.randrange(2, 19) + 0.5)
                                   * height_ratio)
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
