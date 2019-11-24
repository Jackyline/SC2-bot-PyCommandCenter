import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision

import random

from training_data import read_from_file

BATCH_SIZE = 10
EPOCHES = 10
LEARNING_RATE = 0.0001
MOMENTUM = 0.9


# output_classes = ("Offensive", "Defensive")


class StrategyNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_classes):
        super(StrategyNet, self).__init__()
        self.linear1 = nn.Linear(6, 9)
        self.linear2 = nn.Linear(9, 6)
        self.linear3 = nn.Linear(6, 3)

    def forward(self, input):
        output = nn.functional.sigmoid(self.linear1(input))
        output = self.linear2(output)
        output = self.linear3(output)

        return output


net = StrategyNet(6, 10, 8, 3)
data = read_from_file("data.txt")

# Randomize data order
random.shuffle(data)

amount_offensive = 3000
amount_defensive = 0

new_d = []
for d in data:
    if d["strategy"] == "Defensive":
        if amount_defensive > amount_offensive:
            continue
        amount_defensive += 1
        new_d.append(d)
    else:
        new_d.append(d)

print(len(data))
print(len(new_d))

random.shuffle(new_d)
data = new_d
print(len(data))

percent_index = int(len(data) * 0.85)
training_data = data[:percent_index]
testing_data = data[percent_index:]



print(len(training_data))
print(len(testing_data))

import os
#os._exit(1)


criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

# Try add opponent strategy in input (being other than our own)
for epoch in range(EPOCHES):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(training_data, 0):

        state_array = [v for k, v in data["state"].items()]

        inputs = torch.FloatTensor(state_array)

        actual_strategy = data["strategy"]
        if actual_strategy == "Offensive":
            target = torch.FloatTensor([1, 0, 0])
        elif actual_strategy == "Defensive":
            target = torch.FloatTensor([0, 1, 0])
        else:  # actual_strategy == "Expansive":
            target = torch.FloatTensor([0, 0, 1])

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0

print('Finished Training')

""" SAVE MODAL
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
"""


def closest_to(a, b, nr):
    if abs(nr - a) < abs(nr - b):
        return a
    return b


# Test model
correct = 0

offensive = 0
offensive_guessed = 0

defensive = 0
defensive_guessed = 0

expansive = 0
expansive_guessed = 0
for i, data in enumerate(testing_data, 0):

    state_array = [v for k, v in data["state"].items()]

    inputs = torch.FloatTensor(state_array)

    actual_strategy = data["strategy"]

    if actual_strategy == "Offensive":
        target = torch.FloatTensor([1, 0, 0])
        offensive += 1
    elif actual_strategy == "Defensive":
        target = torch.FloatTensor([0, 1, 0])
        defensive += 1
    else:  # actual_strategy == "Expansive":
        target = torch.FloatTensor([0, 0, 1])
        expansive += 1

    output = net(inputs)

    output_list = output.tolist()
    target_list = target.tolist()

    strat = output_list.index(max(output_list))
    if strat == 0:
        offensive_guessed += 1
    if strat == 1:
        defensive_guessed += 1
    if strat == 2:
        expansive_guessed += 1

    #print(output_list, inputs.tolist(), target.tolist())

    # If highest percent class is same as actual class
    if output_list.index(max(output_list)) == target_list.index(max(target_list)):
        correct += 1

print("Percent correct classifications on test data: {}".format(correct / len(testing_data)))
print("offensive: {} out of {}".format(offensive_guessed, offensive))
print("expansive: {} out of {}".format(expansive_guessed, expansive))
print("defensive: {} out of {}".format(defensive_guessed, defensive))

# print(net(torch.FloatTensor([2, 0.45, 7.50])))
