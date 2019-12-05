from classes.task import Task
from classes.task_type import TaskType
from library import *


class TaskManager:
    """
    TODO
    """

    def __init__(self, task_manager):
        self.first = True  # TODO remove
        self.second = False
        self.done = False
        self.task_manager = task_manager

    def on_step(self):
        """
        TODO
        """
        """
        if self.first:
            task1 = Task(TaskType.MINING, Point2DI(7, 7))
            task2 = Task(TaskType.GAS, Point2DI(7, 7))
            self.task_manager.add_task(task1)
            self.task_manager.add_task(task2)
            self.first = False

        if self.second and not self.done:
            task3 = Task(TaskType.BUILD, Point2DI(7, 7))
            self.task_manager.add_task(task3)
            self.second = False
            self.done = True

        self.second = True
        """
    def remove_finished_assignments(self):
        pass

    def mining(self):
        pass
