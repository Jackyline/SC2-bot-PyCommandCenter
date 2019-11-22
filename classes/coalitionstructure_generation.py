#from library import *
from typing import Dict, Any


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

#   def create_coalition(self, type_coalition : {UNIT_TYPEID: int}) -> [[(UNIT_TYPEID, int)], [(UNIT_TYPEID, int)]]:
 #       """"
 #       Input should be a dictionary with UNIT_TYPEID as key and nr of units of that type as the value.
 #       Return is structured as [coalition1, coalition2, ...] where each coalition is a list as
 #       [(UNIT_TYPEID, nr of units), (UNIT_TYPEID, nr of units), ...]
#        """
#
 #       # Ha en koalition vara en lista med antalet av varje typ av agent, d.v.s [nr av typ 1, nr av typ 2, ...]
#
 #       # Initialize type coalition with all units
 #       for i, agent_type in enumerate(type_coalition.keys()):
 #           coalition = [i] = type_coalition[agent_type]
#
 #       self.init_v(coalition)
 #       return None

    def f(self, coalition):
        """
        OBS self.all_b must be updated before calling.
        :param coalition: coalition to genereate optimal coalition structure for
        :return: optimal coalition structure value
        """
        if(sum(coalition) == 0):
            return 0

        # Generate all possible list where every element <= equivalent coalition element
        # Remove last as we dont want [0, 0, ..., 0] list
        all_b = []
        self.generate_all_b(coalition, all_b)
        all_b = all_b[:-1]

        max_value = self.get_v(coalition)
        max_coal = [coalition]
        for set_of_b in all_b:
            # Create new coalition [a1-b1, a2-b2, ...]
            new_col = []
            for i in range(len(set_of_b)):
                new_col.append(coalition[i] - set_of_b[i])

            value = self.get_q(new_col) + self.get_v(set_of_b)
            if value > max_value:
                max_value = value
                max_coal = self.get_r(new_col) + [set_of_b] #this is basically optimal structure for new_col + set of b

        self.set_r(coalition, max_coal)
        return max_value

    def set_q(self, coalition, value):
        if self.coalition_str(coalition) in self.q_dict:
            if value < self.q_dict[self.coalition_str(coalition)]:
                return
        self.q_dict[self.coalition_str(coalition)] = value

    def get_q(self, coalition):
        """
        Get the optimal value for the coalition from Q table if it exists, calculate it otherwise.
        :param coalition: Coalition to get optiamal coalition structure value from
        :return: integer optimal coalition structure value
        """
        if self.coalition_str(coalition) in self.q_dict:
            return self.q_dict[self.coalition_str(coalition)]
        else:
            f_value = self.f(coalition)
            self.q_dict[self.coalition_str(coalition)] = f_value
            return f_value

    def set_r(self, coalition, optimal_cs):
        self.r_dict[self.coalition_str(coalition)] = optimal_cs

    def get_r(self, coalition):
        if self.coalition_str(coalition) in self.r_dict:
            coal_str = self.coalition_str(coalition)
            if self.r_dict[coal_str] == coalition:
                return coalition
            else:
                # input coalition's optimal structure is not just the input coalition so we do:
                # get the optimal structure for the coalition in self.r_dict[coalition] which is the a-b values in definition of f
                # we combine those lists with the optimal structure for coalition - r_dict[coalition] which is the b values in definition of f

                # For every typ of agent, sum all agents of a type in the coalition structure and save in one list
                # e.g. structure [ [1, 2, 2], [1, 3, 5]] would become [2, 5, 7]
                #known_struct = [ sum([temp_coal[agent_type] for temp_coal in self.r_dict[coal_str]]) for agent_type in range(len(coalition))]
                #optiaml_structure = self.r_dict[coal_str] + \
                #       self.get_r([original_coal - optimal_coal for original_coal, optimal_coal in zip(coalition, known_struct)])
                return self.r_dict[coal_str]
        else:
            return None

    def v(self, coalition : []) -> int:
        """
        Utility function, takes in a type coalition and returns a value for that type coalition
        :param coalition: coalition type to be evaluated
        :return: int
        """
        # TODO: Testa v med minsta koalition för olika antal av varje typ t.ex. [5, 3, 3]
        #  bästa borde vara [1, 1, 1], [2, 1, 1], [2, 1, 1]?
        #"""
        if any([nr_of_agents == 0 for nr_of_agents in coalition]):
            return 0
        value = 1 / sum(map(lambda x: 1 + x**2, coalition))
        """
        value = 0
        if coalition.count(0) == 2:
            return sum(map(lambda x: x**2, coalition))
        else:
            return 0
        """
        return value

    def get_v(self, coalition):
        if self.coalition_str(coalition) in self.v_dict:
            return self.v_dict[self.coalition_str(coalition)]
        else:
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
list_to_test = [5, 5, 3]
b = []
a = csg.get_q(list_to_test)
cs = csg.get_r(list_to_test)
print("optimal structure for {} has a value {} and is splitted as:".format(list_to_test, a))
output = ""
for coalition in cs:
    output += "({}, value: {}), ".format(coalition, csg.q_dict[csg.coalition_str(coalition)])
print(output[:-2])