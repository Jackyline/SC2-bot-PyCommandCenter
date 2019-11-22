from library import *
import random


class ScoutUnit:
    def __init__(self, scout_unit):
        self.unit = scout_unit
        self.goal = None
        self.visited = []
        self.frame_stamps = []

    def get_unit(self):
        return self.unit

    def get_goal(self):
        return self.goal

    def get_visited(self):
        return self.visited

    def is_idle(self):
        return self.unit.is_idle

    def is_alive(self):
        return self.unit.is_alive

    def set_goal(self, goal, current_frame):
        if self.goal is not None:
            self.visited.append(self.goal)
            self.frame_stamps.append(current_frame)

        self.unit.move(goal)
        self.goal = goal

    def check_if_visited(self, goals, current_frame):
        for point in goals:
            goal = Point2DI(point[0], point[1])
            if goal not in self.visited:
                self.set_goal(goal, current_frame)
                break
            else:
                index = self.visited.index(goal)
                time_visited = self.frame_stamps[index]
                if goal == self.visited[0] and (current_frame - time_visited) < 200:
                    goal = Point2DI(random.randrange(0, 6), random.randrange(0, 6))
                    self.set_goal(goal, current_frame)
                else:
                    self.set_goal(goal, current_frame)