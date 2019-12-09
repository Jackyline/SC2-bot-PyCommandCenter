from classes import unit_manager, building_manager, scouting_manager, building_strategy
from library import *


class PrintDebug:
    def __init__(self, ida_bot, building_manager: building_manager.BuildingManager,
                 unit_manager: unit_manager.UnitManager, scout_manager: scouting_manager.ScoutingManager,
                 building_strategy: building_strategy.BuildingStrategy, strategy_network, on : bool):
        self.ida_bot = ida_bot
        self.building_manager = building_manager
        self.unit_manager = unit_manager
        self.scout_manager = scout_manager
        self.building_strategy = building_strategy
        self.strategy_network = strategy_network
        self.on = on
        self.print_on_unit = True

    def on_step(self):
        if not self.on:
            return

        # BuildingManager.print_debug returns a string in the format "type: nr\ntype: nr\n"
        # UnitManager.get_info returns a dictionary with two keys, "workerUnits", "militaryUnits" each with a dict
        # of the unit summary as its value.
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

        text = "Buildings: \n_ _ _ _ _ _ _\n\n{}\n Workers: \n_ _ _ _ _ _ _\n\n{}\n Military: \n _ _ _ _ _ _ _\n\n{} \n\n{}".format(
            building_str, worker_str, military_str, self.scout_manager.print_debug())
        self.ida_bot.map_tools.draw_text_screen(0.01, 0.01, text)


        # Building strategy prints:
        build_strat_text = "Building_Strategy: {}".format(self.building_strategy.action())
        self.ida_bot.map_tools.draw_text_screen(0.01, 0.25, build_strat_text)

        # Game strategy prints:
        strategy = self.strategy_network.get_strategy()
        game_strat_text = "Strategy: {}".format(strategy.name)
        self.ida_bot.map_tools.draw_text_screen(0.01, 0.50, game_strat_text)


        # Player base location, used to retrieve mineral fields and geysers.
        base_location = self.ida_bot.base_location_manager.get_player_starting_base_location(
            player_constant=PLAYER_SELF)

        if not self.print_on_unit:
            return

        """ ############# FINNS BUGG HÄR ###########
    
        # Prints unit debug information on each unit
        units = list(self.ida_bot.get_my_units())
        for unit in units:
            # Find what coalition the unit is in (if any) and add that to the print
            coalition = -1
            for i, coalition in enumerate(self.unit_manager.groups):
                if unit in coalition:
                    coalition = i
                    break
            self.ida_bot.map_tools.draw_text(unit.position, " %s id: %d, coal: %d" % (str(unit.unit_type), unit.id, coalition))

        # Print mineral information on each mineral
        minerals = list(base_location.minerals)
        for mineral in minerals:
            self.ida_bot.map_tools.draw_text(mineral.position, " %s id: %d" % (str(mineral.unit_type), mineral.id))

        # Print geyser information on each geyser
        geysers = list(base_location.geysers)
        for geyser in geysers:
            self.ida_bot.map_tools.draw_text(geyser.position, " %s id: %d" % (str(geyser.unit_type), geyser.id))
    
        """
        for worker in self.unit_manager.worker_units:
            task = worker.get_task()
            if task is not None:
                task = task.task_type
            self.ida_bot.map_tools.draw_text(position=worker.get_unit().position, text=str(task))
