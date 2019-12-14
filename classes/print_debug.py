from classes import unit_manager, building_manager, scouting_manager, building_strategy
from library import *


class PrintDebug:
    def __init__(self, ida_bot, building_manager: building_manager.BuildingManager,
                 unit_manager: unit_manager.UnitManager, scout_manager: scouting_manager.ScoutingManager,
                 building_strategy: building_strategy.BuildingStrategy, on : bool):
        self.ida_bot = ida_bot
        self.building_manager = building_manager
        self.unit_manager = unit_manager
        self.scout_manager = scout_manager
        self.building_strategy = building_strategy
        self.on = on
        self.print_on_unit = True

        # Strings saved so we dont need to update info every tick
        self.building_worker_military_text = ""
        self.build_strat_text = ""
        self.game_strat_text = ""
        self.last_guess = None
        self.scout_manager_text = ""

    def on_step(self):
        if not self.on:
            return


        if self.ida_bot.current_frame % 30 == 0:
            building_str = self.building_manager.print_debug()
            units = self.unit_manager.get_info()
            worker_units = units["workerUnits"].items()
            military_units = units["militaryUnits"].items() #TODO: this line crashes when we have multiple unit types
            worker_str = ""
            for type, count in worker_units:
                worker_str += "{}: {}\n".format(type, count)
            military_str = ""
            for type, count in military_units:
                military_str += "{}: {}\n".format(type, count)

            self.building_worker_military_text = "Buildings: \n_ _ _ _ _ _ _\n\n{}\n Workers: \n_ _ _ _ _ _ _\n\n{}\n Military: \n _ _ _ _ _ _ _\n\n{} \n".format(
                building_str, worker_str, military_str)

            # Building strategy prints:
            self.build_strat_text = "Building_Strategy: {}".format(self.building_strategy.last_action)

            # Game strategy prints:
            strategy = self.ida_bot.strategy_network.actual_strategy
            self.last_guess = self.ida_bot.strategy_network.last_res
            self.game_strat_text = "Strategy: {}".format(strategy.name)

            # Player base location, used to retrieve mineral fields and geysers.
            base_location = self.ida_bot.base_location_manager.get_player_starting_base_location(
                player_constant=PLAYER_SELF)

            self.scout_manager_text = self.scout_manager.print_debug()

        self.ida_bot.map_tools.draw_text_screen(0.01, 0.01, self.building_worker_military_text)
        self.ida_bot.map_tools.draw_text_screen(0.01, 0.25, self.build_strat_text)
        self.ida_bot.map_tools.draw_text_screen(0.01, 0.30, self.game_strat_text)
        self.ida_bot.map_tools.draw_text_screen(0.01, 0.35, str(self.last_guess))
        self.ida_bot.map_tools.draw_text_screen(0.01, 0.40, self.scout_manager_text)

        for worker in self.unit_manager.worker_units:
            task = worker.get_task()
            if task is not None:
                task = task.task_type.name
            self.ida_bot.map_tools.draw_text(position=worker.get_unit().position, text=str(task))

        for military_unit in  self.unit_manager.military_units:
            pos = military_unit.get_unit().position
            self.ida_bot.map_tools.draw_text(position=pos, text= "Positon: ("+str(pos.x)+","+str(pos.y)+")")

        if not self.print_on_unit:
            return

        """
        # Prints unit debug information on each unit
        units = list(self.ida_bot.get_my_units())
        for unit in units:
            # Find what coalition the unit is in (if any) and add that to the print
            temp = -1
            if self.unit_manager.groups is not None:
                for i, coalition in enumerate(self.unit_manager.groups):
                    if unit.id in coalition:
                        temp = i
                        break
            self.ida_bot.map_tools.draw_text(unit.position, " %s id: %d, coal: %d" % (str(unit.unit_type), unit.id, temp))


        ############# FINNS BUGG HÄR (kanske) ###########
        # Print mineral information on each mineral
        minerals = list(base_location.minerals)
        for mineral in minerals:
            self.ida_bot.map_tools.draw_text(mineral.position, " %s id: %d" % (str(mineral.unit_type), mineral.id))

        # Print geyser information on each geyser
        geysers = list(base_location.geysers)
        for geyser in geysers:
            self.ida_bot.map_tools.draw_text(geyser.position, " %s id: %d" % (str(geyser.unit_type), geyser.id))
        """
