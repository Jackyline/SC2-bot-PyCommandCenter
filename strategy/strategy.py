import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision

from dummy_data import read_from_file


# output_classes = ("Offensive", "Defensive")


class StrategyNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_classes):
        super(StrategyNet, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_classes)
        self.relu3 = nn.ReLU()

    def forward(self, input):
        output = self.fc1(input)
        # output = F.relu(output)

        output = self.relu1(output)
        output = self.fc2(output)
        output = self.relu2(output)
        # output = F.relu(output)
        output = self.fc3(output)
        #output = self.relu3(output)
        # output = F.relu(output)

        return output


net = StrategyNet(3, 4, 4, 2)

training_data = read_from_file("dummyBadData.txt")
testing_data = read_from_file("dummyTrainingData.txt")

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(training_data, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = torch.FloatTensor(data["input"])
        target = torch.FloatTensor(data["output"])

        if data["output"] == [2]:
            target = torch.FloatTensor([0, 1])
        else:
            target = torch.FloatTensor([1, 0])

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        #print("Input: {}".format(inputs))
        #print("Target: {}".format(target))
        #print("Output: {}".format(outputs))


        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


def closest_to(a, b, nr):
    if abs(nr - a) < abs(nr - b):
        return a
    return b


# Test model
correct = 0
for i, data in enumerate(testing_data, 0):
    inputs = torch.FloatTensor(data["input"])
    target = torch.FloatTensor(data["output"])

    if data["output"] == [2]:
        target = torch.FloatTensor([0, 1])
    else:
        target = torch.FloatTensor([1, 0])

    output = net(inputs)

    output_list = output.tolist()
    target_list = target.tolist()

    # If highest percent class is same as actual class
    if output_list.index(max(output_list)) == target_list.index(max(target_list)):
        # print("{} == {}".format(target_list, output_list))
        correct += 1
    #else:
        #print("{} => {} != {}".format(inputs.tolist(), target_list, output_list))

print("Percent correct classifications on test data: {}".format(correct / len(testing_data)))

print(net(torch.FloatTensor([2, 0.45, 7.50])))
print(net(torch.FloatTensor([2, 0.55, 7.50])))
print(net(torch.FloatTensor([2, 0.60, 7.50])))
print(net(torch.FloatTensor([2, 0.65, 7.50])))

print("")

print(net(torch.FloatTensor([2, 0.60, 2.3])))
print(net(torch.FloatTensor([2, 0.60, 4.2])))
print(net(torch.FloatTensor([2, 0.60, 5.58])))
print(net(torch.FloatTensor([2, 0.60, 7.50])))
print(net(torch.FloatTensor([2, 0.60, 9.90])))


