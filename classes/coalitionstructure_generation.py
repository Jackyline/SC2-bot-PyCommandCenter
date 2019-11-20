from library import *

class CoalitionstructureGeneration:

    def __init__(self):
        #self.idabot = idabot
        self.test_types = ["marine", "tank", "healer"]
        self.v_dict = {}
        self.q_dict = {}
        self.r_dict = {}
        # TODO: byt v_dict mot nested list med max(coalition[i]) rader och en kolumn för varje typ
        self.all_b = []
        # TODO: kan skapa all_b för start koalitionen och sedan bara använda värden som går för mindre koalitioner,
        #  alltså bara värden där alla b < a
        return

    def create_coalition(self, type_coalition : {UNIT_TYPEID: int}) -> [[(UNIT_TYPEID, int)], [(UNIT_TYPEID, int)]]:
        """"
        Input should be a dictionary with UNIT_TYPEID as key and nr of units of that type as the value.
        Return is structured as [coalition1, coalition2, ...] where each coalition is a list as
        [(UNIT_TYPEID, nr of units), (UNIT_TYPEID, nr of units), ...]
        """

        # Ha en koalition vara en lista med antalet av varje typ av agent, d.v.s [nr av typ 1, nr av typ 2, ...]

        # Initialize type coalition with all units
        for i, agent_type in enumerate(type_coalition.keys()):
            coalition = [i] = type_coalition[agent_type]

        self.init_v(coalition)
        return None

    def f(self, coalition):
        """
        OBS self.all_b must be updated before calling.
        :param coalition: coalition to genereate optimal coalition structure for
        :return: optimal coalition structure value
        """
        all_b = []
        self.generate_all_b(coalition, all_b)

        for set_of_b in all_b:
            new_col = []
            for i in range(len(set_of_b)):
                new_col.append(coalition[i] - set_of_b[i])
            value = self.get_q(new_col) + self.get_v(set_of_b)


        self.get_q()

        return None

    def set_q(self, coalition, value):
        if coalition in self.q_dict:
            if value < self.q_dict[self.coalition_str(coalition)]:
                return
        self.q_dict[self.coalition_str(coalition)] = value


    def get_q(self, coalition):
        """
        Get the optimal value for the coalition from Q table if it exists, calculate it otherwise.
        :param coalition: Coalition to get optiamal coalition structure value from
        :return: integer optimal coalition structure value
        """
        try:
            return self.q_dict[self.coalition_str(coalition)]
        except:
            f_value = self.f(coalition)
            self.q_dict[self.coalition_str(coalition)] = f_value
            return f_value

    def get_r(self, coalition):
        try:
            return self.v_dict[self.coalition_str(coalition)]
        except:
            return None

    def v(self, coalition : []) -> int:
        """
        Utility function, takes in a type coalition and returns a value for that type coalition
        :param coalition: coalition type to be evaluated
        :return: int
        """
        # TODO
        return 2 if coalition[0] < 6 else 1

    def get_v(self, coalition):
        try:
            return self.v_dict[self.coalition_str(coalition)]
        except:
            value = self.v(coalition)
            self.v_dict[self.coalition_str(coalition)] = value
            return value

    def init_v(self, coalition, index = 0):
        """
        Calculate value of every possible coalition and save in v_dict.
        :param coalition: coalition type
        :return: None
        """
        if index >= len(coalition):
            self.v_dict[self.coalition_str(coalition)] = self.v(coalition)
            return

        start_value = coalition[index]
        while coalition[index] >= 0:
            self.init_v(coalition, index + 1)
            coalition[index] -= 1
        # Restore the coalition to the same value as it started with
        coalition[index] = start_value

    def coalition_str(self, coalition):
        a = ""
        for x in coalition:
            a += x.__str__() + ","

        # Remove last , and return
        return a[:-1]

    def generate_all_b(self, coalition, all_b, index = 0):
        """
        :param coalition: coalition type
        :return: None
        """
        if index >= len(coalition):
            all_b.append(coalition.copy())
            return

        start_value = coalition[index]
        while coalition[index] >= 0:
            self.generate_all_b(coalition, all_b, index + 1)
            coalition[index] -= 1
        # Restore the coalition to the same value as it started with
        coalition[index] = start_value

csg = CoalitionstructureGeneration()
b = []
csg.generate_all_b([4, 2, 3], b)
print(b)
