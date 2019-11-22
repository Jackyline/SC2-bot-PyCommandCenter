import numpy as np

class QTable:
    def __init__(self, idabot):
        self.file = 'resources/test.npy'
        self.idabot = idabot
        self.q_table = {} # {state (str) : [q_value for action 0 (int), q_value for action 1 (int)]}
        self.read_table()

    def on_step(self):
        if self.idabot.current_frame % 100 == 0:
            print("saved_table")
            print(self.q_table)
            self.save_table()

    def read_table(self):
        try:
            self.q_table = np.load(self.file, allow_pickle=True).item()
        except:
            self.q_table = {}

    def save_table(self):
        np.save(self.file, self.q_table)

    def clear_table(self):
        np.save(self.file, {})

    def get_action(self, key):
        actions = self.q_table.get(key, [0, 0])
        return 0 if actions[0] > actions[1] else 1

    def get_value(self, key, action):
        return self.q_table.get(key, [0, 0])[action]


    def set_value(self, key, action,  value):
        actions = self.q_table.get(key, [0, 0])
        actions[action] = value
        self.q_table[key] = actions

    def on_exit(self):
        self.save_table()
"""
a = QTable(None)
a.clear_table()
"""