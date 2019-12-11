import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from classes.assignment_manager import AssignmentManager
from library import *
from classes.task import Task
from classes.task_type import TaskType
from classes.resource_manager import ResourceManager


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 75)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x


class BuildingStrategy:
    def __init__(self, idabot, resource_manager: ResourceManager, assignment_manager: AssignmentManager):
        self.idabot = idabot
        self.last_build = 0
        self.first = 0
        self.resource_manager = resource_manager
        self.assignment_manager = assignment_manager
        self.observations = []
        self.model = Net()
        self.model.load_state_dict(torch.load('./buildingstrategy/model_final.pth'))
        self.model.eval()
        self.last_action = ""
        # TODO fix this in other file
        self.actions = {}
        for key, value in action_id.items():
            self.actions[value] = key


    def update_obs(self, observations):
        pass

    def create_task(self):
        pass

    def action(self):

        gas = self.resource_manager.get_gas()
        minerals = self.resource_manager.get_minerals()
        # If we have enough resources, produce some important tasks that our network won't predict too often
        if self.idabot.current_frame % 100 == 0 and minerals > 400 and gas > 100:

            # Marauder, siege tank, hellion, techlab
            wanted_units = []

            curr_seconds = self.idabot.current_frame // 24
            if curr_seconds > 30:
                wanted_units = [*wanted_units,
                                self.name_to_type("Marauder"),
                                self.name_to_type("Marauder"),
                                self.name_to_type("Marauder"),
                                self.name_to_type("Marauder"),
                                self.name_to_type("Marauder"),
                ]
            if curr_seconds > 30 and len(self.idabot.building_manager.get_buildings_of_type(self.name_to_type("CommandCenter"))) <= 3:
                wanted_units = [*wanted_units,
                    self.name_to_type("CommandCenter"),
                    self.name_to_type("CommandCenter"),
                    self.name_to_type("CommandCenter"),
                    self.name_to_type("CommandCenter"),
                    self.name_to_type("CommandCenter"),
                    self.name_to_type("CommandCenter")
                ]

            if curr_seconds > 180:  # After 3 mins, can predict to build any of these
                wanted_units = [*wanted_units,
                                self.name_to_type("SiegeTank"),
                                self.name_to_type("SiegeTank"),
                                self.name_to_type("SiegeTank"),
                                self.name_to_type("Hellion"),
                                self.name_to_type("Hellion"),
                                self.name_to_type("Hellion"),
                                self.name_to_type("TechLab")
                                ]

            if wanted_units:
                index = random.randint(0, len(wanted_units) - 1)

                task = wanted_units[index]
                self.add_task(task)
                self.last_action = task
                return task


        curr_seconds = self.idabot.current_frame // 24
        if curr_seconds - self.last_build < 2:
            return self.last_action

        self.last_build = curr_seconds

        supply = self.resource_manager.get_supply()
        max_supply = self.resource_manager.get_max_supply()

        # normalize
        food_cap = max_supply / 200
        input_minerals = minerals / 62500
        input_gas = gas / 62500
        food_used = supply / 200
        # TODO, fix this
        food_army = len(self.idabot.unit_manager.military_units) / 200
        food_workers = len(self.idabot.unit_manager.worker_units) / 200

        v = torch.Tensor([input_minerals, input_gas, food_cap, food_used, food_army, food_workers])
        # Hardcoded input atm
        predicted = self.model(v)
        predicted_top_three = np.argpartition(predicted.detach().numpy(), -3)[-3:]
        random_choice = random.randrange(1, 100)
        if random_choice <= 70:
            action = predicted_top_three[2]
        elif random_choice <= 90:
            action = predicted_top_three[1]
        else:
            action = predicted_top_three[0]

        action = action_name[self.actions[str(action)]]
        action_type = self.name_to_type(action)

        self.add_task(action_type)
        self.last_action = action
        return action

    def add_task(self, action_type):
        built_prerequisites = self.get_built_prerequisites(action_type)
        if built_prerequisites is not None:
            self.add_task(built_prerequisites)
        if self.resource_manager.can_afford(action_type):
            location_near = self.idabot.base_location_manager.get_player_starting_base_location(
                PLAYER_SELF).depot_position
            if action_type.unit_typeid == UNIT_TYPEID.TERRAN_COMMANDCENTER:
                build_pos = self.idabot.base_location_manager.get_next_expansion(PLAYER_SELF).depot_position
                build_location = Point2D(build_pos.x, build_pos.y)
            else:
                build_pos = self.idabot.building_placer.get_build_location_near(location_near, action_type)
                build_location = Point2D(build_pos.x, build_pos.y)
            task = None

            if action_type.is_worker or action_type.is_combat_unit:
                task = Task(TaskType.TRAIN, produce_unit=action_type)
            elif action_type.is_addon:
                action_type = self.get_random_techlab()
                task = Task(TaskType.ADD_ON, construct_building=action_type)
            elif action_type.is_refinery:
                for manager in self.idabot.base_location_manager.get_occupied_base_locations(PLAYER_SELF):
                    for geyser in manager.geysers:
                        if self.get_refinery(geyser) is None:
                            task = Task(TaskType.BUILD, pos=geyser.position, geyser=geyser,
                                        construct_building=action_type)
                            print("Adding Task: ", task.task_type, "Action_type: ", action_type)
                            self.assignment_manager.add_task(task)
                            return
                return
            elif action_type.is_building:
                task = Task(TaskType.BUILD, pos=build_location, build_position=build_pos,
                            construct_building=action_type)

            print("Adding Task: ", task.task_type, "Action_type: ", action_type)
            self.assignment_manager.add_task(task)

    def get_random_techlab(self):
        random_choice = random.randrange(1, 100)
        if random_choice <= 70:
            return UnitType(UNIT_TYPEID.TERRAN_BARRACKSTECHLAB, self.idabot)
        elif random_choice <= 90:
            return UnitType(UNIT_TYPEID.TERRAN_FACTORYTECHLAB, self.idabot)
        return UnitType(UNIT_TYPEID.TERRAN_STARPORTTECHLAB, self.idabot)

    def get_built_prerequisites(self, action_type):

        # Special case, sometimes get unknown action_types
        if action_type.unit_typeid == UNIT_TYPEID.INVALID:
            return None

        requirement_building_type = UnitType(action_type.required_structure, self.idabot)
        buildings = self.idabot.building_manager.buildings
        our_building_types = [building.get_unit_type() for building in buildings]

        if action_type.unit_typeid == UNIT_TYPEID.TERRAN_TECHLAB:
            action_type = self.get_random_techlab()

        if action_type.is_building or action_type.is_addon:
            return requirement_building_type if requirement_building_type not in our_building_types else None
        elif action_type.is_worker or action_type.is_combat_unit:
            type_data = self.idabot.tech_tree.get_data(action_type)
            what_builds = type_data.what_builds
            return what_builds[0] if not any(
                building_type in our_building_types for building_type in what_builds) else None
        return None

    def get_refinery(self, geyser: Unit):
        """ Returns: A refinery which is on top of unit `geyser` if any, None otherwise """

        def squared_distance(p1: Point2D, p2: Point2D) -> float:
            return (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2

        for unit in self.idabot.get_my_units():
            if unit.unit_type.is_refinery and squared_distance(unit.position, geyser.position) < 1:
                return unit

        return None

    def name_to_type(self, name):
        to_type = {
            "BarracksReactor": UnitType(UNIT_TYPEID.TERRAN_BARRACKSREACTOR, self.idabot),
            "FactoryFlying": UnitType(UNIT_TYPEID.TERRAN_FACTORYFLYING, self.idabot),
            "SupplyDepot": UnitType(UNIT_TYPEID.TERRAN_SUPPLYDEPOT, self.idabot),
            "BarracksTechLab": UnitType(UNIT_TYPEID.TERRAN_BARRACKSTECHLAB, self.idabot),
            "OrbitalCommand": UnitType(UNIT_TYPEID.TERRAN_ORBITALCOMMAND, self.idabot),
            "EngineeringBay": UnitType(UNIT_TYPEID.TERRAN_ENGINEERINGBAY, self.idabot),
            "Bunker": UnitType(UNIT_TYPEID.TERRAN_BUNKER, self.idabot),
            "StarportReactor": UnitType(UNIT_TYPEID.TERRAN_STARPORTREACTOR, self.idabot),
            "Starport": UnitType(UNIT_TYPEID.TERRAN_STARPORT, self.idabot),
            "StarportTechLab": UnitType(UNIT_TYPEID.TERRAN_STARPORTTECHLAB, self.idabot),
            "FusionCore": UnitType(UNIT_TYPEID.TERRAN_FUSIONCORE, self.idabot),
            "MissileTurret": UnitType(UNIT_TYPEID.TERRAN_MISSILETURRET, self.idabot),
            "Factory": UnitType(UNIT_TYPEID.TERRAN_FACTORY, self.idabot),
            "FactoryReactor": UnitType(UNIT_TYPEID.TERRAN_FACTORYREACTOR, self.idabot),
            "Armory": UnitType(UNIT_TYPEID.TERRAN_ARMORY, self.idabot),
            "BarracksFlying": UnitType(UNIT_TYPEID.TERRAN_BARRACKSFLYING, self.idabot),
            "TechLab": UnitType(UNIT_TYPEID.TERRAN_TECHLAB, self.idabot),
            "OrbitalCommandFlying": UnitType(UNIT_TYPEID.TERRAN_ORBITALCOMMANDFLYING, self.idabot),
            "FactoryTechLab": UnitType(UNIT_TYPEID.TERRAN_FACTORYTECHLAB, self.idabot),
            "SensorTower": UnitType(UNIT_TYPEID.TERRAN_SENSORTOWER, self.idabot),
            "CommandCenterFlying": UnitType(UNIT_TYPEID.TERRAN_COMMANDCENTERFLYING, self.idabot),
            "CommandCenter": UnitType(UNIT_TYPEID.TERRAN_COMMANDCENTER, self.idabot),
            "GhostAcademy": UnitType(UNIT_TYPEID.TERRAN_GHOSTACADEMY, self.idabot),
            "PlanetaryFortress": UnitType(UNIT_TYPEID.TERRAN_PLANETARYFORTRESS, self.idabot),
            "Reactor": UnitType(UNIT_TYPEID.TERRAN_REACTOR, self.idabot),
            "Barracks": UnitType(UNIT_TYPEID.TERRAN_BARRACKS, self.idabot),
            "SupplyDepotLowered": UnitType(UNIT_TYPEID.TERRAN_SUPPLYDEPOTLOWERED, self.idabot),
            "AutoTurret": UnitType(UNIT_TYPEID.TERRAN_AUTOTURRET, self.idabot),
            "MULE": UnitType(UNIT_TYPEID.TERRAN_MULE, self.idabot),
            "Medivac": UnitType(UNIT_TYPEID.TERRAN_MEDIVAC, self.idabot),
            "Thor": UnitType(UNIT_TYPEID.TERRAN_THOR, self.idabot),
            "Marauder": UnitType(UNIT_TYPEID.TERRAN_MARAUDER, self.idabot),
            "Battlecruiser": UnitType(UNIT_TYPEID.TERRAN_BATTLECRUISER, self.idabot),
            "Reaper": UnitType(UNIT_TYPEID.TERRAN_REAPER, self.idabot),
            "WidowMine": UnitType(UNIT_TYPEID.TERRAN_WIDOWMINE, self.idabot),
            "Hellion": UnitType(UNIT_TYPEID.TERRAN_HELLION, self.idabot),
            "Raven": UnitType(UNIT_TYPEID.TERRAN_RAVEN, self.idabot),
            "Marine": UnitType(UNIT_TYPEID.TERRAN_MARINE, self.idabot),
            "SiegeTank": UnitType(UNIT_TYPEID.TERRAN_SIEGETANK, self.idabot),
            "SCV": UnitType(UNIT_TYPEID.TERRAN_SCV, self.idabot),
            "VikingFighter": UnitType(UNIT_TYPEID.TERRAN_VIKINGFIGHTER, self.idabot),
            "Cyclone": UnitType(UNIT_TYPEID.TERRAN_CYCLONE, self.idabot),
            "Liberator": UnitType(UNIT_TYPEID.TERRAN_LIBERATOR, self.idabot),
            "Banshee": UnitType(UNIT_TYPEID.TERRAN_BANSHEE, self.idabot),
            "Ghost": UnitType(UNIT_TYPEID.TERRAN_GHOST, self.idabot),
            "Refinery": UnitType(UNIT_TYPEID.TERRAN_REFINERY, self.idabot)
        }

        return to_type[name]


# TODO fix this in other file
action_id = {
    '352': '58',
    '353': '55',
    '354': '57',
    '355': '56',
    '361': '61',
    '362': '63',
    '363': '65',
    '369': '67',
    '370': '70',
    '371': '69',
    '375': '72',
    '378': '73',
    '39': '10',
    '402': '2',
    '403': '3',
    '405': '4',
    '406': '5',
    '410': '6',
    '414': '7',
    '418': '8',
    '419': '9',
    '42': '13',
    '423': '12',
    '43': '14',
    '44': '15',
    '453': '34',
    '459': '39',
    '460': '40',
    '464': '42',
    '468': '44',
    '469': '45',
    '470': '46',
    '475': '54',
    '476': '49',
    '477': '51',
    '478': '52',
    '487': '59',
    '488': '60',
    '490': '62',
    '492': '64',
    '496': '66',
    '498': '68',
    '50': '20',
    '502': '71',
    '53': '21',
    '56': '24',
    '58': '25',
    '64': '29',
    '66': '32',
    '71': '36',
    '72': '37',
    '79': '41',
    '83': '43',
    '89': '47',
    '91': '48',
    '92': '50',
    '93': '53'}
action_name = {
    '352': 'Research_AdvancedBallistics',
    '353': 'Research_BansheeCloakingField',
    '354': 'Research_BansheeHyperflightRotors',
    '355': 'Research_BattlecruiserWeaponRefit',
    '361': 'Research_CombatShield',
    '362': 'Research_ConcussiveShells',
    '363': 'Research_DrillingClaws',
    '369': 'Research_HiSecAutoTracking',
    '370': 'Research_HighCapacityFuelTanks',
    '371': 'Research_InfernalPreigniter',
    '375': 'Research_NeosteelFrame',
    '378': 'Research_PersonalCloaking',
    '39': 'Armory',
    '402': 'Research_RavenCorvidReactor',
    '403': 'Research_RavenRecalibratedExplosives',
    '405': 'Research_Stimpack',
    '406': 'Research_TerranInfantryArmor',
    '410': 'Research_TerranInfantryWeapons',
    '414': 'Research_TerranShipWeapons',
    '418': 'Research_TerranStructureArmorUpgrade',
    '419': 'Research_TerranVehicleAndShipPlating',
    '42': 'Barracks',
    '423': 'Research_TerranVehicleWeapons',
    '43': 'Bunker',
    '44': 'CommandCenter',
    '453': 'Stop',
    '459': 'Banshee',
    '460': 'Battlecruiser',
    '464': 'Cyclone',
    '468': 'Ghost',
    '469': 'Hellbat',
    '470': 'Hellion',
    '475': 'Liberator',
    '476': 'Marauder',
    '477': 'Marine',
    '478': 'Medivac',
    '487': 'Raven',
    '488': 'Reaper',
    '490': 'SCV',
    '492': 'SiegeTank',
    '496': 'Thor',
    '498': 'VikingFighter',
    '50': 'EngineeringBay',
    '502': 'WidowMine',
    '53': 'Factory',
    '56': 'FusionCore',
    '58': 'GhostAcademy',
    '64': 'MissileTurret',
    '66': 'Nuke',
    '71': 'Reactor',
    '72': 'Reactor',
    '79': 'Refinery',
    '83': 'SensorTower',
    '89': 'Starport',
    '91': 'SupplyDepot',
    '92': 'TechLab',
    '93': 'TechLab'}
