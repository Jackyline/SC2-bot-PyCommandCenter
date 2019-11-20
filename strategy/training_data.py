import pickle
import mpyq
import os


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

from s2protocol import versions


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

import sc2reader


def worker_counter(replay, second, player_id):
    workers = []
    for event in replay.events:
        if event.name == "UnitBornEvent" and event.control_pid == player_id:
            if event.unit.is_worker:
                workers.append(event.unit)

        if event.name == "UnitDiedEvent":
            if event.unit in workers:
                workers.remove(event.unit)

        if event.second > second:
            break

    return len(workers)


def army_counter(replay, second, player_id):
    armie = []
    for event in replay.events:
        if event.name == "UnitBornEvent" and event.control_pid == player_id:
            if event.unit.is_army:
                armie.append(event.unit)
                # print(event.unit.name)

        if event.name == "UnitDiedEvent":
            if event.unit in armie:
                armie.remove(event.unit)

        if event.second > second:
            break

    return len(armie)


def building_counter(replay, second, player_id):
    buildings = []
    for event in replay.events:
        if event.name == "UnitDoneEvent" and event.unit.is_building:
            if event.unit.owner.pid == player_id:
                buildings.append(event.unit)
        elif event.name == "UnitDiedEvent" and event.unit.is_building:
            if event.unit.owner.pid == player_id:
                buildings.remove(event.unit)

        if event.second > second:
            break

    return len(buildings)

"""
TargetPointCommandEvent - BuildCommandCenter (EXPAND)

"""


def open_replay2(replay_name):
    # TODO: make this modular to work on all computers, by finding the dir it was started in
    #path = os.path.join(os.listdir("replays"), replay_name)
    replay = sc2reader.load_replay(
        'replays_p3/{filename}'.format(filename=replay_name),
        load_map=True, load_level=4)


    event_names = set([event.name for event in replay.events])
    for elem in event_names:
        print("{} ::: {}".format(elem, elem))

    """
    events_of_type = {name: [] for name in event_names}
    for event in replay.events:
        events_of_type[event.name].append(event)

    for event in events_of_type:
        print(event)

    # for elem in replay.events:
    #    print(elem)
    """
    print("Workers")
    print(worker_counter(replay, 400, 1))
    print(worker_counter(replay, 400, 2))

    print("Armies")
    print(army_counter(replay, 400, 1))
    print(army_counter(replay, 400, 2))

    print("Buildings")
    print(building_counter(replay, 200, 1))
    print(building_counter(replay, 200, 2))

    length_of_game = replay.frames // 24

    print(length_of_game)


open_replay2('Clem_v_Ziggy:_Game_2_-_Kairos_Junction_LE.SC2Replay')
