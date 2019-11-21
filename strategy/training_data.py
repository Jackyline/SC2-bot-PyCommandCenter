import pickle
import mpyq
import os

import sc2reader

from s2protocol import versions

ALL_BUILDINGS = [
    "Refinery"
    "SupplyDepotLowered",
    "BarracksReactor",
    "FactoryFlying",
    "SupplyDepot",
    "BarracksTechLab",
    "OrbitalCommand",
    "BarracksTechLab",
    "EngineeringBay",
    "Bunker",
    "StarportReactor",
    "Starport",
    "StarportTechLab",
    "FusionCore",
    "MissileTurret",
    "Factory",
    "FactoryReactor",
    "Armory",
    "BarracksFlying",
    "TechLab",
    "OrbitalCommandFlying",
    "FactoryTechLab",
    "SensorTower",
    "CommandCenterFlying",
    "CommandCenter",
    "GhostAcademy",
    "PlanetaryFortress",
    "Reactor",
    "Barracks",
    "SupplyDepotLowered"
]

# All possible states of a terran command center
COMMAND_CENTERS = ["CommandCenterFlying",
                   "CommandCenter",
                   "OrbitalCommandFlying",
                   "OrbitalCommand"]


class MatchStates:
    def __init__(self, filename):
        self.states = []
        self.filename = filename

    def save_all_states(self):
        with open(self.filename, "wb") as file:
            pickle.dump(self.states, file)

    def get_all_states(self):
        with open(self.filename, "rb") as file:
            return pickle.load(file)

    def add_match_state(self):
        state = MatchState()
        self.states.append(state)


class MatchState:
    def __init__(self):
        # Observed game state
        self.minerals = None  # [0, ..., n]
        self.gas = None  # [0, .., n]
        self.expansions = None  # [0, ..., n]
        self.armies = None  # [0, ..., n]
        self.defensive_buildings = None  # [0,1]
        self.production_buildings = None  # [0,1]

        self.match_stage = None  # {early, mid, late}

        self.observed_armies = None  # [0,1]
        self.observed_defensive = None  # [0,1]
        self.observed_strategy = None  # [0,1] (None = 0.25 , Offensive: 0.50, Defensive: 0.75, Expansive: 1}

        # Classified from state
        self.classified_strategy = None  # {Offensive, Defensive or Expansive}

    def get_short_parameters_1(self):
        return [self.armies, self.match_stage]

    def get_parameters(self):
        return [self.resources, self.expansions, self.armies, self.defensive_buildings, self.production_buildings,
                self.match_stage, self.observed_armies, self.observed_defensive, self.observed_strategy]

    def get_output(self):
        return self.classified_strategy


def save_states(filename, data):
    with open(filename, "wb") as file:
        pickle.dump(data, file)


def get_states(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


# he = MatchStates("hejsan.txt")


# Extract data from replays

"""
# With SC2Protocol
def open_replay(replay_name):
    # Open replay
    archive = mpyq.MPQArchive(
        '/home/hugo/LIU/tddd92-projekt-2019-storgrupp-2-01/strategy/replays_p3/{filename}'.format(filename=replay_name))

    print(archive.files)

    contents = archive.header['user_data_header']['content']
    header = versions.latest().decode_replay_header(contents)
    baseBuild = header['m_version']['m_baseBuild']
    protocol = versions.build(baseBuild)

    contents = archive.read_file('replay.tracker.events')
    gameEvents = protocol.decode_replay_tracker_events(contents)

    print(gameEvents)

    for event in gameEvents:
        print(event)


# open_replay('Clem_v_Ziggy:_Game_2_-_Kairos_Junction_LE.SC2Replay')
"""


def worker_counter(replay, second, player_id):
    return len(get_units(replay, second, player_id, is_worker_unit))


def army_counter(replay, second, player_id):
    return len(get_units(replay, second, player_id, is_army_unit))


def is_army_unit(unit):
    return unit.is_army


def is_worker_unit(unit):
    return unit.is_worker


def get_units(replay, second, player_id, type_function):
    units = []
    for event in replay.events:
        if event.name == "UnitBornEvent" and event.control_pid == player_id and type_function(event.unit):
            units.append(event.unit)
        elif event.name == "UnitDiedEvent" and event.unit in units:
            units.remove(event.unit)

        if event.second > second:
            break

    return units


def building_counter(replay, second, player_id):
    return len(buildings_of_type(replay, second, player_id, ALL_BUILDINGS))


def amount_expansions(replay, second, player_id):
    return len(buildings_of_type(replay, second, player_id, COMMAND_CENTERS))


def buildings_of_type(replay, second, player_id, types):
    """
    :param replay: Replay file
    :param second: Time from start in seconds
    :param player_id:
    :param types: unit type-names
    :return: All buildings for given player that are given types

    :type replay: SC2REPLAY
    :type second: int
    :type player_id: int
    :type types: list
    """
    buildings = []
    for event in replay.events:
        if event.name == "UnitDoneEvent" and event.unit.is_building and event.unit.owner.pid == player_id \
                and event.unit.name in types:
            buildings.append(event.unit)
        elif event.name == "UnitDiedEvent" and event.unit.is_building and event.unit.owner.pid == player_id \
                and event.unit in buildings:
            buildings.remove(event.unit)

        # Only look up to given time
        if event.second > second:
            break

    return buildings


def is_offensive():
    pass


def is_expansive():
    pass


def get_current_strategy(events):
    # Offensive
    if is_offensive():
        return
    # Expansive
    elif is_expansive():
        return
    # Defensive
    else:
        return


"""
TargetPointCommandEvent - BuildCommandCenter (EXPAND)

"""


def open_replay2(replay_name):
    # TODO: make this modular to work on all computers, by finding the dir it was started in
    # path = os.path.join(os.listdir("replays"), replay_name)
    replay = sc2reader.load_replay(
        'replays_p3/{filename}'.format(filename=replay_name),
        load_map=True, load_level=4)

    """ Print all different types of events
    event_names = set([event.name for event in replay.events])
    for elem in event_names:
        print("{} ::: {}".format(elem, elem))
    """

    for elem in replay.events:
        print(elem)

    return

    print("Workers")
    print(worker_counter(replay, 400, 1))
    print(worker_counter(replay, 400, 2))

    print("Armies")
    print(army_counter(replay, 400, 1))
    print(army_counter(replay, 400, 2))

    print("Buildings")
    print(building_counter(replay, 600, 1))
    print(building_counter(replay, 600, 2))

    print("EXPANSIONS")
    print(amount_expansions(replay, 600, 1))
    print(amount_expansions(replay, 600, 2))

    length_of_game = replay.frames // 24

    print(length_of_game)


open_replay2('Clem_v_Ziggy:_Game_2_-_Kairos_Junction_LE.SC2Replay')
