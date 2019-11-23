import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision

import random

from training_data import read_from_file

BATCH_SIZE = 10

# output_classes = ("Offensive", "Defensive")


class StrategyNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_classes):
        super(StrategyNet, self).__init__()
        self.linear1 = nn.Linear(3, 8)
        self.linear2 = nn.Linear(8, 8)
        self.linear3 = nn.Linear(8, 3)
        self.linear4 = nn.Linear(20, 20)
        self.linear5 = nn.Linear(20, 3)

    def forward(self, input):
        output = F.sigmoid(self.linear1(input))
        output = F.sigmoid(self.linear2(output))
        output = F.sigmoid(self.linear3(output))
        #output = F.sigmoid(self.linear4(output))
        #output = F.sigmoid(self.linear5(output))

        return output


net = StrategyNet(6, 10, 8, 3)
data = read_from_file("data.txt")
#random.shuffle(data)

print(len(data))

training_data = data[:10000]
"""
train_d = []
for j in range(10000//BATCH_SIZE):
    batch = []
    for i in range(BATCH_SIZE):
        batch.append(training_data[i])
    train_d.append(batch)
for elem in train_d:
    print(elem)
"""
testing_data = data[10000:11000]

criterion = nn.MSELoss()
# NLLLoss
# KLDivLoss
# MarginRankingLoss
# HingeEmbeddingLoss
# CosineEmbeddingLoss
# CrossEntropyLoss

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

for epoch in range(3):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(training_data, 0):

        state = data["state"]
        state_array = [state["workers"],
                       state["armies"],
                       state["minerals"],
                       state["vespene"],
                       state["expansions"],
                       state["time"],
                       ]

        input = torch.FloatTensor(state_array)

        actual_strategy = data["strategy"]
        if actual_strategy == "Offensive":
            target = torch.FloatTensor([1, 0, 0])
        elif actual_strategy == "Defensive":
            target = torch.FloatTensor([0, 1, 0])
        elif actual_strategy == "Expansive":
            target = torch.FloatTensor([0, 0, 1])

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(input)

        loss = criterion(outputs, target)
        #loss = F.cross_entropy(outputs.float(), target.long().view(3))

        loss.backward()

        optimizer.step()

        # print("Input: {}".format(inputs))
        # print("Target: {}".format(target))
        # print("Output: {}".format(outputs))

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

    state = data["state"]
    state_array = [state["workers"],
                   state["armies"],
                   state["minerals"],
                   state["vespene"],
                   state["expansions"],
                   state["time"],
                   ]

    input = torch.FloatTensor(state_array)

    actual_strategy = data["strategy"]

    if actual_strategy == "Offensive":
        target = torch.FloatTensor([1, 0, 0])
        offensive += 1
    elif actual_strategy == "Defensive":
        target = torch.FloatTensor([0, 1, 0])
        defensive += 1
    elif actual_strategy == "Expansive":
        target = torch.FloatTensor([0, 0, 1])
        expansive += 1

    output = net(input)

    output_list = output.tolist()
    target_list = target.tolist()

    strat = output_list.index(max(output_list))
    if strat == 0:
        offensive_guessed += 1
    if strat == 1:
        defensive_guessed += 1
    if strat == 2:
        expansive_guessed += 1

    print(output_list, input.tolist(), target.tolist())

    # If highest percent class is same as actual class
    if output_list.index(max(output_list)) == target_list.index(max(target_list)):
        # print("{} == {}".format(target_list, output_list))

        correct += 1
    # else:
    # print("{} => {} != {}".format(inputs.tolist(), target_list, output_list))

print("Percent correct classifications on test data: {}".format(correct / len(testing_data)))
print("offensive: {} out of {}".format(offensive_guessed, offensive))
print("expansive: {} out of {}".format(expansive_guessed, expansive))
print("defensive: {} out of {}".format(defensive_guessed, defensive))

# print(net(torch.FloatTensor([2, 0.45, 7.50])))
