import torch
import torch.nn as nn
import torch.nn.functional as F
from classes.resource_manager import ResourceManager

actions = { 76: "SCV" }

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 2)
        self.fc2 = nn.Linear(2, 78)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class BuildingStrategy:
    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager
        self.observations = []
        self.model = Net()
        self.model.load_state_dict(torch.load('./buildingstrategy/model.pth'))
        self.model.eval()


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

        print("gas: {}".format(gas))
        print("minerals: {}".format(minerals))
        print("supply: {}".format(supply))
        print("supply: {}".format(max_supply))

        # Hardcoded input atm
        input = torch.tensor([1]).float()
        _, predicted = torch.max(self.model(input), 0)
        return actions[predicted.item()]