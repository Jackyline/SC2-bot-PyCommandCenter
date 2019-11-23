import pickle
import mpyq
import os
import math

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


def formatReplay(replay):
    return """
    {filename}
    --------------------------------------------
    SC2 Version {release_string}
    {category} Game, {start_time}
    {type} on {map_name}
    Length: {game_length}
    
    """.format(**replay.__dict__)


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
        if event.name in ["UnitBornEvent", "UnitBornEvent"] and event.control_pid == player_id and type_function(
                event.unit):
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
        # Only look up to given time
        if event.second > second:
            break
        # UnitBornEvent
        if event.name in ["UnitDoneEvent",
                          "UnitBornEvent"] and event.unit.is_building and event.unit.owner.pid == player_id \
                and event.unit.name in types:
            buildings.append(event.unit)
        elif event.name == "UnitDiedEvent" and event.unit.is_building and event.unit.owner.pid == player_id \
                and event.unit in buildings:
            buildings.remove(event.unit)

    return buildings


"""
Could look on these:
UnitPositionUpdate (see which units were hardmed in last 15 seconds)
Ability Attack (See when a player attacks the opponent)

"""


def max_distance_between(attack_location, base_locations):
    """
    :param attack_location: (x,y) cordinates
    :param base_locations:  [(x,y), ..]
    :return: Maximum distance from :param attack_location to any
    of bases in :param base_locations
    """
    max_distance = 0
    math.sqrt(math.pow(attack_location[0] - base_locations[0][0], 2))
    for base in base_locations:
        distance = math.sqrt(math.pow(attack_location[0] - base[0], 2) +
                             math.pow(attack_location[1] - base[1], 2))
        max_distance = distance if distance > max_distance else max_distance
    return max_distance


def is_offensive(replay, second, player, time_offset):
    attack_event = None

    for event in replay.events:
        if event.name == "TargetPointCommandEvent" and event.ability_name == "Attack" \
                and event.pid == player and event.second < second:
            attack_event = event

    # Not offensive if no attack or attack is too long ago
    if not attack_event or math.fabs(attack_event.second - second) > time_offset:
        return False

    select_event = None
    for event in replay.events:
        if event.name == "SelectionEvent" and event.pid == player \
                and event.second < attack_event.second:
            select_event = event

    # Not offensive if not units were selected to attack with
    if not select_event:
        return False

    # Selected units that are also army units
    armies_selected = list(filter(lambda x: x.is_army, select_event.new_units))

    # Not offensive if no armies were moved
    if not armies_selected:
        return False

    attack_location = attack_event.location

    # Location of players command centers
    base_locations = [com.location for com in buildings_of_type(replay, second, player, COMMAND_CENTERS)]

    # Only class as attack if outside of command centers
    if max_distance_between(attack_location, base_locations) > 30:
        print(armies_selected)
        return True

    return False


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
PlayerstatsEvent - minerals_current, vespene_current, stats
UnitBornEvent - location, x, y, unit_id
"""


def get_current_minerals(replay, second, player_id):
    return get_resource_information(replay, second, player_id, lambda x: x.minerals_current)


def get_current_vespene(replay, second, player_id):
    return get_resource_information(replay, second, player_id, lambda x: x.vespence_current)


def get_resource_information(replay, second, player_id, filter_function):
    closest_event = None
    for event in replay.events:
        if event.name != "PlayerStatsEvent" or event.pid != player_id:
            continue
        elif event.second > second:
            break
        else:
            closest_event = event

    return filter_function(closest_event) if closest_event else 0


def get_position_of_armies(replay, second, player_id):
    units = get_units(replay, second, player_id, is_army_unit)
    return {unit.id: unit.location for unit in units}


def get_event_types(replay):
    """ Get all different types of events """
    return set([event.name for event in replay.events])


def open_replay2(replay_name):
    # Open replay file and make it into an object
    """
    for replay in sc2reader.load_replays("replays_p3"):
        print(formatReplay(replay))
    """

    replay = sc2reader.load_replay(
        'replays_p3/{filename}'.format(filename=replay_name),
        load_map=True, load_level=4)
    # Print general match info
    print(formatReplay(replay))

    for event in get_event_types(replay):
        print(event)

    for i in range(1000):
        print("Offensive at {}: {}".format(i, is_offensive(replay, i, 1, 15)))
    return

    print(is_offensive(replay, 500, 1))
    print(is_offensive(replay, 500, 1))
    print(amount_expansions(replay, 500, 1))

    print(worker_counter(replay, 10, 1))

    for elem in replay.events:
        print(elem)
    return
    """ 
    TODO: Skulle kunna kolla på Ability - Attack för att se när ngn är offensiv, samt
    när det byggs ny expansion.
    
    Kolla på senaste selectionEvent samt efterkommande rightClick.
    
    Vad är 'Right Click' för class?
    """
    # print(get_event_types(replay))

    # print(replay.players[0].attribute_data)
    # print(replay.players[0].detail_data)
    # print(replay.players[0].units)

    secs = 400

    loc1 = get_position_of_armies(replay, secs, 2)
    loc11 = get_position_of_armies(replay, secs + 30, 2)
    loc2 = get_position_of_armies(replay, secs, 2)

    print(get_position_of_armies(replay, secs, 2))

    # print(loc1)
    # print(loc11)
    return
    print(army_counter(replay, secs, 1))
    print(loc2)
    print(army_counter(replay, secs, 2))

    """
    UnitTypeChangeEvent, 
    ?? UpdateTargetUnitCommandEvent, 
    """

    return

    for unit in replay.players[0].units:
        if unit.name == "Marine":
            print(unit)
            print(type(unit))
            print(unit.location)
            break

    return

    counter = 500

    for a, b in get_position_of_armies(replay, 1000, 1).items():
        print("{} : {}".format(a, b))

    print("Armies")
    print(len(get_position_of_armies(replay, 500, 1)))
    # print(army_counter(replay, 500, 1))

    return

    print("{} moved out of {}".format(len(get_position_of_armies(replay, counter, 1)),
                                      army_counter(replay, counter, 1)))
    print("{} moved out of {}".format(len(get_position_of_armies(replay, counter, 2)),
                                      army_counter(replay, counter, 2)))

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
    print(amount_expansions(replay, 650, 1))
    print(amount_expansions(replay, 650, 2))

    length_of_game = replay.frames // 24

    print(length_of_game)


open_replay2('Clem_v_Ziggy:_Game_2_-_Kairos_Junction_LE.SC2Replay')

# This is not working as intended, since replays don't contain the information
# wanted about positions of units.
"""
# TODO: go up to given second and update all units positions
# Not working :(, unit locations not being updated as the game goes.
def get_position_of_units(replay, second, player_id, filter_function=None):
    unit_positions = {}
    for event in sorted(replay.events, key=lambda x: x.second):

        if event.name == "UnitPositionsEvent":
            if event.second > second:
                continue
            # print(event)
            # print(event.second)
            # print(event.units)
            # Find the latest position update
            for unit, pos in event.units.items():
                if unit.owner.pid == player_id and (not filter_function or filter_function(unit)):
                    unit_positions[unit] = pos


        elif event.name == "UnitDiedEvent" and event.unit.is_army and event.unit.owner.pid == player_id \
                and event.unit in unit_positions:
            del unit_positions[event.unit]

            # elif event.name == "UnitDiedEvent" and event.unit in unit_positions:
            #    del unit_positions[event.unit]

    unit_positions2 = {}
    for event in sorted(replay.events, key=lambda x: x.second):

        if event.name == "UnitPositionsEvent":
            if event.second > second:
                continue
            # Find the latest position update
            print(event.positions)
            print(len(event.positions))
            print(event.units)
            print(len(event.units))
            print(event.items)
            print(len(event.items))
            print(army_counter(replay, event.second, player_id))
            continue
            for index, pos in event.units.items():
                if unit.owner.pid == player_id and (not filter_function or filter_function(unit)):
                    unit_positions[unit.pid] = pos

    return unit_positions

    
        latest_event = None
        for event in replay.events:

            if event.name == "UnitPositionsEvent":
                # Find the latest position update
                if event.second > second:
                    break
                latest_event = event

    return {unit: pos for unit, pos in latest_event.units.items()
                if unit.owner.pid == player_id and (not filter_function or filter_function(unit))}\
        if latest_event else []
    """
