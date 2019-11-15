import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dummy_data import read_from_file

output_classes = ("Offensive", "Defensive")


class StrategyNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_classes):
        super(StrategyNet, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_classes)

    def forward(self, input):
        output = self.fc1(input)
        output = self.relu1(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output = self.fc3(output)
        return output


net = StrategyNet(2, 6, 4, 2)

"""
print(net)

params = list(net.parameters())
print(len(params))

input = torch.randn(2)
output = net(input)
print(input)
print(output)"""

training_data = read_from_file()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(training_data, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = torch.FloatTensor(data["input"])
        #labels = torch.FloatTensor(data["output"])

        if data["output"] == [2]:
            labels = torch.FloatTensor([0, 1])
        else:
            labels = torch.FloatTensor([1, 0])

        inputs = torch.randn(1, 2)
        labels = torch.randn(1, 2)

        print(inputs.size())
        print(labels.size())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        print(outputs.size())
        loss = criterion(outputs, labels)
        #loss.backward()
        #optimizer.step()

        # print statistics
        #running_loss += loss.item()
        if i % 200 == 199:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Finished Training')
