import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
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
        self.model.load_state_dict(torch.load('./buildingstrategy/model_3_epochs.pth'))
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
        action = random.choice(predicted_top_three)

        return action_name[self.actions[str(action)]]

# TODO fix this in other file
action_id = {'140': '1',
             '168': '11',
             '261': '0',
             '300': '16',
             '301': '17',
             '304': '18',
             '305': '19',
             '309': '22',
             '312': '23',
             '317': '26',
             '318': '27',
             '319': '28',
             '320': '33',
             '321': '30',
             '322': '31',
             '326': '35',
             '327': '38',
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
action_name = {'140': 'Cancel_quick',
               '168': 'Cancel_Last_quick',
               '261': 'Halt_quick',
               '300': 'Morph_Hellbat_quick',
               '301': 'Morph_Hellion_quick',
               '304': 'Morph_LiberatorAAMode_quick',
               '305': 'Morph_LiberatorAGMode_screen',
               '309': 'Morph_OrbitalCommand_quick',
               '312': 'Morph_PlanetaryFortress_quick',
               '317': 'Morph_SiegeMode_quick',
               '318': 'Morph_SupplyDepot_Lower_quick',
               '319': 'Morph_SupplyDepot_Raise_quick',
               '320': 'Morph_ThorExplosiveMode_quick',
               '321': 'Morph_ThorHighImpactMode_quick',
               '322': 'Morph_Unsiege_quick',
               '326': 'Morph_VikingAssaultMode_quick',
               '327': 'Morph_VikingFighterMode_quick',
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
               '42': 'Build_Barracks_screen',
               '423': 'Research_TerranVehicleWeapons_quick',
               '43': 'Build_Bunker_screen',
               '44': 'Build_CommandCenter_screen',
               '453': 'Stop_quick',
               '459': 'Train_Banshee_quick',
               '460': 'Train_Battlecruiser_quick',
               '464': 'Train_Cyclone_quick',
               '468': 'Train_Ghost_quick',
               '469': 'Train_Hellbat_quick',
               '470': 'Train_Hellion_quick',
               '475': 'Train_Liberator_quick',
               '476': 'Train_Marauder_quick',
               '477': 'Train_Marine_quick',
               '478': 'Train_Medivac_quick',
               '487': 'Train_Raven_quick',
               '488': 'Train_Reaper_quick',
               '490': 'Train_SCV_quick',
               '492': 'Train_SiegeTank_quick',
               '496': 'Train_Thor_quick',
               '498': 'Train_VikingFighter_quick',
               '50': 'Build_EngineeringBay_screen',
               '502': 'Train_WidowMine_quick',
               '53': 'Build_Factory_screen',
               '56': 'Build_FusionCore_screen',
               '58': 'Build_GhostAcademy_screen',
               '64': 'Build_MissileTurret_screen',
               '66': 'Build_Nuke_quick',
               '71': 'Build_Reactor_quick',
               '72': 'Build_Reactor_screen',
               '79': 'Build_Refinery_screen',
               '83': 'Build_SensorTower_screen',
               '89': 'Build_Starport_screen',
               '91': 'Build_SupplyDepot_screen',
               '92': 'Build_TechLab_quick',
               '93': 'Build_TechLab_screen'}