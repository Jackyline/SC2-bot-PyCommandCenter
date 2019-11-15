import numpy as np

class QTable:
    def __init__(self, idabot):
        self.idabot = idabot
        self.q_table = {}
        self.read_table()

    def on_step(self):
        if self.idabot.current_frame % 500 == 0:
            self.save_table()

    def read_table(self):
        self.q_table = np.load('resources/test.npy', allow_pickle=True).item()

    def save_table(self):
        np.save('resources/test.npy', self.q_table)

    def get(self, key):
        return self.q_table.get(key, 0)

    def set(self, key, value):
        self.q_table[key] = value

    def on_exit(self):
        self.save_table()