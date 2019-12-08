from classes.task_type import TaskType
from library import *


class Task:
    """
    Ni kan skapa en Task och sedan köra köra add_task(task) i assignment_manager så kommer uppgiften automatiskt tilldelas
    När ni skapar en task, ange alltid vilken typ av task det är med TaskType.TYPE_OF_task samt:

    attack: ange position för attack
    defend: ange position för defend
    mining: ange position för command center i den bas jobbet ska utföras
    gas: ange position på refinery
    build: ange vad som ska byggas samt position?
    train: ange vad son ska tränas, kan ange position om vi listar ut vart vi vill ha enhheten

    så tex Task(TaskType.ATTACK, position)
    """

    def __init__(self, task_type: TaskType, pos: Unit.tile_position = None, construct_building=None, produce_unit=None):
        self.task_type = task_type
        self.pos = pos
        self.construct_building = construct_building
        self.produce_unit = produce_unit
