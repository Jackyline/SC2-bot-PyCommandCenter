import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision

import random

from training_data import read_from_file

BATCH_SIZE = 10
EPOCHES = 5
LEARNING_RATE = 0.00001
MOMENTUM = 0.9
DATA_FILE = "data.txt"
MODAL_NAME = "strategy/network"


# output_classes = ("Offensive", "Defensive")


class StrategyNet(nn.Module):
    def __init__(self):
        super(StrategyNet, self).__init__()
        self.linear1 = nn.Linear(49, 32)
        self.linear2 = nn.Linear(32, 24)
        self.linear3 = nn.Linear(24, 16)
        self.linear4 = nn.Linear(16, 8)
        self.linear5 = nn.Linear(8, 2)

    def forward(self, input):
        output = torch.relu(self.linear1(input))
        output = torch.relu(self.linear2(output))
        output = torch.relu(self.linear3(output))
        output = torch.relu(self.linear4(output))
        output = self.linear5(output)

        return output


class StrategyNetwork():
    def __init__(self):
        self.net = StrategyNet()

    def save_network(self, filename):
        if not self.net:
            raise Exception("No network to save")

        torch.save(self.net.state_dict(), "{}".format(filename))

    def load_network(self, filename):
        self.net.load_state_dict(torch.load("{}".format(filename)))
        self.net.eval()

    def calculate(self, inputs):
        '''
        :return: The output of the network from given :param inputs
        '''
        return self.net(torch.FloatTensor(inputs)).tolist()

    def train_network(self, training_data):
        #criterion = nn.MSELoss()
        #criterion = F.cross_entropy()
        optimizer = optim.SGD(self.net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

        # Try add opponent strategy in input (being other than our own)
        for epoch in range(EPOCHES):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(training_data, 0):

                state_array = [v for k, v in sorted(data["state"].items(), key=lambda x: x[0])]

                inputs = torch.FloatTensor(state_array)

                actual_strategy = data["strategy"]
                if actual_strategy == "Offensive":
                    target = torch.LongTensor([0])
                else:  # actual_strategy == "Defensive":
                    target = torch.LongTensor([1])
                    #target = torch.autograd.variable()
                    #target = torch.autograd.Variable(torch.randn((3, 5)))
                #target = torch.empty(1, dtype=torch.long).random_(2)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)

                outputs = outputs.view(1, 2)


                #loss = criterion(outputs, target)
                loss = F.cross_entropy(outputs, target)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 1000 == 999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 1000))
                    running_loss = 0.0

        print('Finished Training')

    def test_network(self, testing_data):
        # Test model
        correct = 0

        offensive = 0
        offensive_guessed = 0
        correct_offensive_guessed = 0

        defensive = 0
        defensive_guessed = 0
        correct_defensive_guessed = 0

        for i, data in enumerate(testing_data, 0):

            # state_array = [v for k, v in data["state"].items()]
            state_array = [v for k, v in sorted(data["state"].items(), key=lambda x: x[0])]

            inputs = torch.FloatTensor(state_array)

            actual_strategy = data["strategy"]

            if actual_strategy == "Offensive":
                target = torch.FloatTensor([1, 0])
                offensive += 1
            else:  #actual_strategy == "Defensive":
                target = torch.FloatTensor([0, 1])
                defensive += 1

            output = self.net(inputs)

            output_list = output.tolist()
            target_list = target.tolist()

            network_guessed_strategy = output_list.index(max(output_list))
            # Guessed Offensive
            if network_guessed_strategy == 0:
                offensive_guessed += 1
                if actual_strategy == "Offensive":
                    correct_offensive_guessed += 1
            # Guessed Defensive
            elif network_guessed_strategy == 1:
                defensive_guessed += 1
                if actual_strategy == "Defensive":
                    correct_defensive_guessed += 1

            # print(output_list, inputs.tolist(), target.tolist())

            # If highest percent class is same as actual class
            if output_list.index(max(output_list)) == target_list.index(max(target_list)):
                correct += 1

        print("Percent correct classifications on test data: {}".format(correct / len(testing_data)))
        print("offensive: {} out of {}. {} of them were correct.".format(offensive_guessed, offensive,
                                                                         correct_offensive_guessed))
        print("defensive: {} out of {}. {} of them were correct.".format(defensive_guessed, defensive,
                                                                         correct_defensive_guessed))


def get_data():
    # Load training data
    data = read_from_file(DATA_FILE)

    # Randomize data order
    random.shuffle(data)

    # Use same amount of data points for each strategy
    amount_offensive = 11000
    amount_defensive = 0

    new_d = []
    for d in data:
        # To balance amount of all states
        if d["strategy"] == "Defensive":
            if amount_defensive > amount_offensive:
                continue
            amount_defensive += 1
            new_d.append(d)
        else:
            new_d.append(d)

    random.shuffle(new_d)
    data = new_d

    print("Amount of data points: {}".format(len(data)))

    percent_index = int(len(data) * 0.85)
    training_data = data[:percent_index]
    testing_data = data[percent_index:]

    return training_data, testing_data


def get_trained_network():
    net = StrategyNetwork()
    net.load_network(MODAL_NAME)
    return net


def create_network():
    training_data, testing_data = get_data()

    net = StrategyNetwork()
    net.train_network(training_data)
    net.test_network(testing_data)

    #net.save_network(MODAL_NAME)

create_network()
# n = net.load_network("network")
# net.test_network(testing_data)

""" SAVE MODAL
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
"""

# print(net(torch.FloatTensor([2, 0.45, 7.50])))
