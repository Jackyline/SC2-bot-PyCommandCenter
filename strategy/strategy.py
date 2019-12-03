import strategy_model as strategy_model


class Strategy():
    def __init__(self):
        self.model = strategy_model.get_trained_network()

    def get_strategy(self, inputs):
        res = self.model.calculate(inputs)

        strategy = res.index(max(res))
        if strategy == 0:
            return "Offensive"
        elif strategy == 1:
            return "Defensive"
        else:
            return "Expansive"

"""
a = Strategy()
import random

d = {}
for i in range(50000):
    result = a.get_strategy([random.randint(0, 80),
                             random.randint(0, 100),
                             random.randint(0, 500),
                             random.randint(0, 500),
                             random.randint(1, 6),
                             random.randint(0, 25)])
    if result in d:
        d[result] += 1
    else :
        d[result] = 1

# result = a.model.calculate([90,29,378,59,2,15])

print(d)

"""
