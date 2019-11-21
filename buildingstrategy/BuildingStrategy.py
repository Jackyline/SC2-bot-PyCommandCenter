import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self):
        self.observations = []

        self.model = Net()
        self.model.load_state_dict(torch.load('model.pth'))
        self.model.eval()


    def update_obs(self, observations):
        pass

    """
        Returns an action.
    """
    def action(self):
        # Hardcoded input atm
        input = torch.tensor([1]).float()
        _, predicted = torch.max(self.model(input), 0)
        return actions[predicted.item()]