import strategy.strategy_model as strategy_model
from library import UnitType, UNIT_TYPEID
from strategy.training_data import ALL_BUILDINGS, UNIT_TYPES
from enum import Enum


class StrategyName(Enum):
    OFFENSIVE = 0
    DEFENSIVE = 1


# Wait this many seconds if strategy is OFFENSIVE and new strategy is DEFENSIVE, this is done since OFFENSIVE
# takes some time to actually perform, and we don't want to walk back and forth all the time.
STRATEGY_DELAY = 20


class Strategy():
    def __init__(self, idabot):
        self.model = strategy_model.get_trained_network()
        self.idabot = idabot
        self.actual_strategy = StrategyName.DEFENSIVE  # (Strategy, Time)
        self.last_updated_strategy = 0
        self.last_res = [0, 0]

    # TODO: Might want to "randomize" OFFENSIVE strategy sometimes if it never gets predicted.
    def get_strategy(self):
        # Inputs to model
        inputs = self.get_strategy_inputs()

        # Output from model
        res = self.model.calculate(inputs)

        self.last_res = res

        # Get new predicted strategy, OFFENSIVE or DEFENSIVE
        new_strategy = StrategyName(res.index(max(res)))

        # Current game seconds
        curr_seconds = self.idabot.current_frame // 24

        # Requires between 10 and 40 military units to attack
        military_required_to_attack = min(max((curr_seconds * 2) // 60, 10), 40)

        # Should not be allowed to be offensive if not enough military units
        if not len(self.idabot.unit_manager.military_units) > military_required_to_attack:
            new_strategy = StrategyName.DEFENSIVE


        # Update our strategy to OFFENSIVE, if network predicts so and we have enough military units.
        if new_strategy == StrategyName.OFFENSIVE:
            self.actual_strategy = StrategyName.OFFENSIVE
            self.last_updated_strategy = curr_seconds
        # If last strategy was OFFENSIVE, we want to wait STRATEGY_DELAY seconds before changing it to DEFENSIVE
        elif self.actual_strategy == StrategyName.OFFENSIVE and new_strategy == StrategyName.DEFENSIVE and \
                curr_seconds - self.last_updated_strategy >= STRATEGY_DELAY:
            self.actual_strategy = StrategyName.DEFENSIVE
            self.last_updated_strategy = curr_seconds

        return self.actual_strategy

    def get_strategy_inputs(self):
        unit_type_to_name = {
            UnitType(UNIT_TYPEID.TERRAN_BARRACKSREACTOR, self.idabot): "BarracksReactor",
            UnitType(UNIT_TYPEID.TERRAN_FACTORYFLYING, self.idabot): "FactoryFlying",
            UnitType(UNIT_TYPEID.TERRAN_SUPPLYDEPOT, self.idabot): "SupplyDepot",
            UnitType(UNIT_TYPEID.TERRAN_BARRACKSTECHLAB, self.idabot): "BarracksTechLab",
            UnitType(UNIT_TYPEID.TERRAN_ORBITALCOMMAND, self.idabot): "OrbitalCommand",
            UnitType(UNIT_TYPEID.TERRAN_ENGINEERINGBAY, self.idabot): "EngineeringBay",
            UnitType(UNIT_TYPEID.TERRAN_BUNKER, self.idabot): "Bunker",
            UnitType(UNIT_TYPEID.TERRAN_STARPORTREACTOR, self.idabot): "StarportReactor",
            UnitType(UNIT_TYPEID.TERRAN_STARPORT, self.idabot): "Starport",
            UnitType(UNIT_TYPEID.TERRAN_STARPORTTECHLAB, self.idabot): "StarportTechLab",
            UnitType(UNIT_TYPEID.TERRAN_FUSIONCORE, self.idabot): "FusionCore",
            UnitType(UNIT_TYPEID.TERRAN_MISSILETURRET, self.idabot): "MissileTurret",
            UnitType(UNIT_TYPEID.TERRAN_FACTORY, self.idabot): "Factory",
            UnitType(UNIT_TYPEID.TERRAN_FACTORYREACTOR, self.idabot): "FactoryReactor",
            UnitType(UNIT_TYPEID.TERRAN_ARMORY, self.idabot): "Armory",
            UnitType(UNIT_TYPEID.TERRAN_BARRACKSFLYING, self.idabot): "BarracksFlying",
            UnitType(UNIT_TYPEID.TERRAN_TECHLAB, self.idabot): "TechLab",
            UnitType(UNIT_TYPEID.TERRAN_ORBITALCOMMANDFLYING, self.idabot): "OrbitalCommandFlying",
            UnitType(UNIT_TYPEID.TERRAN_FACTORYTECHLAB, self.idabot): "FactoryTechLab",
            UnitType(UNIT_TYPEID.TERRAN_LIBERATORAG, self.idabot): "RefinerySupplyDepotLowered",  # TODO: Fix this one!
            UnitType(UNIT_TYPEID.TERRAN_SENSORTOWER, self.idabot): "SensorTower",
            UnitType(UNIT_TYPEID.TERRAN_COMMANDCENTERFLYING, self.idabot): "CommandCenterFlying",
            UnitType(UNIT_TYPEID.TERRAN_COMMANDCENTER, self.idabot): "CommandCenter",
            UnitType(UNIT_TYPEID.TERRAN_GHOSTACADEMY, self.idabot): "GhostAcademy",
            UnitType(UNIT_TYPEID.TERRAN_PLANETARYFORTRESS, self.idabot): "PlanetaryFortress",
            UnitType(UNIT_TYPEID.TERRAN_REACTOR, self.idabot): "Reactor",
            UnitType(UNIT_TYPEID.TERRAN_BARRACKS, self.idabot): "Barracks",
            UnitType(UNIT_TYPEID.TERRAN_SUPPLYDEPOTLOWERED, self.idabot): "SupplyDepotLowered",
            UnitType(UNIT_TYPEID.TERRAN_AUTOTURRET, self.idabot): "AutoTurret",
            UnitType(UNIT_TYPEID.TERRAN_MULE, self.idabot): "MULE",
            UnitType(UNIT_TYPEID.TERRAN_MEDIVAC, self.idabot): "Medivac",
            UnitType(UNIT_TYPEID.TERRAN_THOR, self.idabot): "Thor",
            UnitType(UNIT_TYPEID.TERRAN_MARAUDER, self.idabot): "Marauder",
            UnitType(UNIT_TYPEID.TERRAN_BATTLECRUISER, self.idabot): "Battlecruiser",
            UnitType(UNIT_TYPEID.TERRAN_REAPER, self.idabot): "Reaper",
            UnitType(UNIT_TYPEID.TERRAN_WIDOWMINE, self.idabot): "WidowMine",
            UnitType(UNIT_TYPEID.TERRAN_HELLION, self.idabot): "Hellion",
            UnitType(UNIT_TYPEID.TERRAN_RAVEN, self.idabot): "Raven",
            UnitType(UNIT_TYPEID.TERRAN_MARINE, self.idabot): "Marine",
            UnitType(UNIT_TYPEID.TERRAN_SIEGETANK, self.idabot): "SiegeTank",
            UnitType(UNIT_TYPEID.TERRAN_SCV, self.idabot): "SCV",
            UnitType(UNIT_TYPEID.TERRAN_VIKINGFIGHTER, self.idabot): "VikingFighter",
            UnitType(UNIT_TYPEID.TERRAN_CYCLONE, self.idabot): "Cyclone",
            UnitType(UNIT_TYPEID.TERRAN_LIBERATOR, self.idabot): "Liberator",
            UnitType(UNIT_TYPEID.TERRAN_BANSHEE, self.idabot): "Banshee",
            UnitType(UNIT_TYPEID.TERRAN_GHOST, self.idabot): "Ghost",
        }

        units = {unit: 0 for unit in [*ALL_BUILDINGS, *UNIT_TYPES]}

        # Count amount of all units
        for unit in self.idabot.get_my_units():
            if unit.unit_type in unit_type_to_name:
                units[unit_type_to_name[unit.unit_type]] += 1

        # Is this how you get the actual seconds?
        curr_seconds = self.idabot.current_frame // 24
        inputs = {
            "minerals": self.idabot.resource_manager.resources.minerals,
            "vespene": self.idabot.resource_manager.resources.gas,
            "time": curr_seconds
        }

        state = {**inputs, **units}

        # Insert all values depending in their sorted key order
        state_array = [value for key, value in sorted(state.items(), key=lambda x: x[0])]

        return state_array


"""
a = Strategy()
import random

d = {}
for i in range(50000):
    result = a.get_strategy([random.randint(0, 80),
                             random.randint(0, 100),
                             random.randint(0, 500),
                             random.randint(0, 500),
                             random.randint(1, 6),
                             random.randint(0, 25)])
    if result in d:
        d[result] += 1
    else :
        d[result] = 1

# result = a.model.calculate([90,29,378,59,2,15])

print(d)

"""
