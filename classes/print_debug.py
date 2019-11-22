from classes import unit_manager, building_manager, scouting_manager
from library import *


class PrintDebug:
    def __init__(self, ida_bot, building_manager: building_manager.BuildingManager,
                 unit_manager: unit_manager.UnitManager, scout_manager: scouting_manager.ScoutingManager, on : bool):
        self.ida_bot = ida_bot
        self.building_manager = building_manager
        self.unit_manager = unit_manager
        self.scout_manager = scout_manager
        self.on = on
        self.print_on_unit = False

    def on_step(self):
        if not self.on:
            return

        # BuildingManager.print_debug returns a string in the format "type: nr\ntype: nr\n"
        # UnitManager.get_info returns a dictionary with two keys, "workerUnits", "militaryUnits" each with a dict
        # of the unit summary as its value.
        building_str = self.building_manager.print_debug()
        units = self.unit_manager.get_info()
        worker_units = sorted(units["workerUnits"].items(), key=(lambda x: x[0]))
        military_units = sorted(units["militaryUnits"].items(), key=(lambda x: x[0]))
        worker_str = ""
        for type, count in worker_units:
            worker_str += "{}: {}\n".format(type, count)
        military_str = ""
        for type, count in military_units:
            military_str += "{}: {}\n".format(type, count)

        text = "Buildings: \n_ _ _ _ _ _ _\n\n{}\n Workers: \n_ _ _ _ _ _ _\n\n{}\n Military: \n _ _ _ _ _ _ _\n\n{} \n\n{}".format(
            building_str, worker_str, military_str, self.scout_manager.print_debug())
        self.ida_bot.map_tools.draw_text_screen(0.01, 0.01, text)

        # Player base location, used to retrieve mineral fields and geysers.
        base_location = self.ida_bot.base_location_manager.get_player_starting_base_location(
            player_constant=PLAYER_SELF)

        if not self.print_on_unit:
            return

        # Prints unit debug information on each unit
        units = list(self.ida_bot.get_my_units())
        for unit in units:
            self.ida_bot.map_tools.draw_text(unit.position, " %s id: %d" % (str(unit.unit_type), unit.id))

        # Print mineral information on each mineral
        minerals = list(base_location.minerals)
        for mineral in minerals:
            self.ida_bot.map_tools.draw_text(mineral.position, " %s id: %d" % (str(mineral.unit_type), mineral.id))

        # Print geyser information on each geyser
        geysers = list(base_location.geysers)
        for geyser in geysers:
            self.ida_bot.map_tools.draw_text(geyser.position, " %s id: %d" % (str(geyser.unit_type), geyser.id))
