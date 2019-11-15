# !!!!!!!!!!!!!! Create really bad training data just for my own sake, this file will probably be deleted. !!!!!!!!!!!!!!
# from strategy.training_data import MatchState
import random
import json

AMOUNT = 1000

output = []

for i in range(AMOUNT):
    armie = round(random.uniform(0, 1), 1)
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

    output.append({"input": [match_stage, armie], "output": [strategy]})


def write_to_file(data):
    with open("dummyBadData.txt", "w") as file:
        file.write(json.dumps(data, indent=4))


def read_from_file():
    with open("dummyBadData.txt", "r") as file:
        return json.loads(file.read())


#write_to_file(output)


print(read_from_file())