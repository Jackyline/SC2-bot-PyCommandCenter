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
        

    """
        Returns an action.
    """
    def get_action():
        pass