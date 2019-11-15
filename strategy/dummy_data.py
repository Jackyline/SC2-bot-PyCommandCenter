# !!!!!!!!!!!!!! Create really bad training data just for my own sake, this file will probably be deleted. !!!!!!!!!!!!!!
# from strategy.training_data import MatchState
import random
import json

def random_data(amount):
    output = []
    for i in range(amount):
        armie = round(random.uniform(0, 1), 2)
        match_stage = random.randint(1, 3)

        strategy = 0  # Defensive
        if armie > 0.8 and match_stage == 3:
            strategy = 1  # Offensive
        elif armie > 0.6 and match_stage == 2:
            strategy = 1  # Offensive
        elif armie > 0.4 and match_stage == 1:
            strategy = 1  # Offensive
        else:
            strategy = 2  # Defensive

        random_numer = round(random.uniform(0, 10),1)

        output.append({"input": [match_stage, armie, random_numer], "output": [strategy]})
    return output


def write_to_file(data, filename):
    with open(filename, "w") as file:
        file.write(json.dumps(data, indent=4))


def read_from_file(filename):
    with open(filename, "r") as file:
        return json.loads(file.read())


write_to_file(random_data(10000), "dummyBadData.txt")

write_to_file(random_data(1000), "dummyTrainingData.txt")
