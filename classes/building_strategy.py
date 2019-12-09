import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from library import UnitType, UNIT_TYPEID
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
    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager
        self.observations = []
        self.model = Net()
        self.model.load_state_dict(torch.load('./buildingstrategy/model_final.pth'))
        self.model.eval()
        #TODO fix this in other file
        self.actions = {}
        for key, value in action_id.items():
            self.actions[value] = key


    def update_obs(self, observations):
        pass

    """
        Returns an action.
    """
    def action(self):
        gas = self.resource_manager.get_gas()
        minerals = self.resource_manager.get_minerals()
        supply = self.resource_manager.get_supply()
        max_supply = self.resource_manager.get_max_supply()

        #normalize
        food_cap = max_supply / 200
        input_minerals = minerals/ 62500
        input_gas = gas / 62500
        food_used = supply / 200
        # TODO, fix this
        food_army = (supply - supply/2) / 200
        food_workers = (supply / 2) / 200
        """
        print("gas: {}".format(gas))
        print("minerals: {}".format(minerals))
        print("supply: {}".format(supply))
        print("supply: {}".format(max_supply))
        """
        v = torch.Tensor([input_minerals, input_gas, food_cap, food_used, food_army, food_workers])
        # Hardcoded input atm
        predicted = self.model(v)
        predicted_top_three = np.argpartition(predicted.detach().numpy(), -3)[-3:]
        random_choice = random.randrange(1,100)
        if random_choice <= 70:
            action = predicted_top_three[2]
        elif random_choice <= 90:
            action = predicted_top_three[1]
        else:
            action = predicted_top_three[0]

        return action_name[self.actions[str(action)]]

    def bajs(self):
        name_to_type = {
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
        }

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
               '352': 'Research_AdvancedBallistics_quick',
               '353': 'Research_BansheeCloakingField_quick',
               '354': 'Research_BansheeHyperflightRotors_quick',
               '355': 'Research_BattlecruiserWeaponRefit_quick',
               '361': 'Research_CombatShield_quick',
               '362': 'Research_ConcussiveShells_quick',
               '363': 'Research_DrillingClaws_quick',
               '369': 'Research_HiSecAutoTracking_quick',
               '370': 'Research_HighCapacityFuelTanks_quick',
               '371': 'Research_InfernalPreigniter_quick',
               '375': 'Research_NeosteelFrame_quick',
               '378': 'Research_PersonalCloaking_quick',
               '39': 'Build_Armory_screen',
               '402': 'Research_RavenCorvidReactor_quick',
               '403': 'Research_RavenRecalibratedExplosives_quick',
               '405': 'Research_Stimpack_quick',
               '406': 'Research_TerranInfantryArmor_quick',
               '410': 'Research_TerranInfantryWeapons_quick',
               '414': 'Research_TerranShipWeapons_quick',
               '418': 'Research_TerranStructureArmorUpgrade_quick',
               '419': 'Research_TerranVehicleAndShipPlating_quick',
               '42': 'Barracks',
               '423': 'Research_TerranVehicleWeapons_quick',
               '43': 'Build_Bunker_screen',
               '44': 'CommandCenter',
               '453': 'Stop_quick',
               '459': 'Train_Banshee_quick',
               '460': 'Train_Battlecruiser_quick',
               '464': 'Train_Cyclone_quick',
               '468': 'Train_Ghost_quick',
               '469': 'Train_Hellbat_quick',
               '470': 'Train_Hellion_quick',
               '475': 'Train_Liberator_quick',
               '476': 'Train_Marauder_quick',
               '477': 'Marine',
               '478': 'Train_Medivac_quick',
               '487': 'Train_Raven_quick',
               '488': 'Train_Reaper_quick',
               '490': 'SCV',
               '492': 'Train_SiegeTank_quick',
               '496': 'Train_Thor_quick',
               '498': 'Train_VikingFighter_quick',
               '50': 'Build_EngineeringBay_screen',
               '502': 'Train_WidowMine_quick',
               '53': 'Factory',
               '56': 'Build_FusionCore_screen',
               '58': 'Build_GhostAcademy_screen',
               '64': 'Build_MissileTurret_screen',
               '66': 'Build_Nuke_quick',
               '71': 'Build_Reactor_quick',
               '72': 'Build_Reactor_screen',
               '79': 'RefineryFBar',
               '83': 'Build_SensorTower_screen',
               '89': 'Build_Starport_screen',
               '91': 'SupplyDepot',
               '92': 'Build_TechLab_quick',
               '93': 'Build_TechLab_screen'}