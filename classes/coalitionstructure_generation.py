from library import *
from typing import Dict, Any, List


class CoalitionstructureGeneration:
    """
    Nästan gjorde:
    Integrera med boten, var ska funktionen kallas? ska csg klassen fördela specifika enheter? etc.

    Att göra (TODO):
    Kom på sätt att spara v, q, r listor så att det är mycket snabbare, blir fett weird med nestade listor
    Fixa så att vi kan skapa ett max antal koalitioner
    Effektivisera all_b delen? Kör all_b på start koalitionen och kolla på delmängder av den när vi kollar på mindre
        koalitioner?
    Värderings funktion!!
    """

    def __init__(self):
        # self.idabot = idabot
        self.test_types = ["marine", "tank", "healer"]
        self.v_dict = {}
        self.q_dict = {}
        self.r_dict = {}
        self.all_b = []
        return

    #    def create_coalition(self, military_units: Dict[UNIT_TYPEID: List[Unit.id]]) -> List[List[Unit.id]]:
    def create_coalition(self, military_units):
        """"
        Input should be a dictionary with UNIT_TYPEID as key and a list of unit ids of all units of that type as the
        value. Return is structured as [coalition1, coalition2, ...] where each coalition is a list as
        [[UnitId, unitId]], (UNIT_TYPEID, nr of units), ...]
        """

        # Ha en koalition vara en lista med antalet av varje typ av agent, d.v.s [nr av typ 1, nr av typ 2, ...]

        # Initialize type coalition with all units
        coalition = []
        for agent_count in military_units.values():
            coalition.append(len(agent_count))

        self.f(coalition)
        cs = self.get_r(coalition)

        # Create the output coalition structure by designating a unit (by id) for each unit the coalition should have.
        output_coalition_structure = []
        agent_types = list(military_units.keys())
        for coalition in cs:
            output_coalition = []
            # For each type of agent in a coalition
            for current_agent_type in range(len(coalition)):
                # Insert as many units of type current_agent_type as the coalition has
                for agent_count in range(coalition[current_agent_type]):
                    output_coalition.append(military_units[agent_types[current_agent_type]].pop(-1))
            output_coalition_structure.append(output_coalition)

        return output_coalition_structure

    def f(self, coalition):
        """
        Calculate the optimal value of the input coalition. The coalition structure can be fetched with get_r function
        :param coalition: coalition to genereate optimal coalition structure for
        :return: optimal coalition structure value
        """
        if (sum(coalition) == 0):
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
                max_coal = self.get_r(new_col) + [
                    set_of_b]  # this is basically optimal structure for new_col + set of b

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
                return self.r_dict[coal_str]
        else:
            return None

    def v(self, coalition: []) -> int:
        """
        Utility function, takes in a type coalition and returns a value for that type coalition
        :param coalition: coalition type to be evaluated
        :return: int
        """
        # TODO: Testa v med minsta koalition för olika antal av varje typ t.ex. [5, 3, 3]
        #  bästa borde vara [1, 1, 1], [2, 1, 1], [2, 1, 1]?
        # """
        #if len(coalition) - coalition.count(0) != 1:
        #    return 0
        #value = sum(map(lambda x: 1 + x ** 2, coalition))
        #"""
        value = 0
        if coalition.count(0) > 0:
            return 0
        else:
            return 1 / sum(map(lambda x: x ** 2, coalition))
        #"""
        return value

    def get_v(self, coalition):
        if self.coalition_str(coalition) in self.v_dict:
            return self.v_dict[self.coalition_str(coalition)]
        else:
            value = self.v(coalition)
            self.v_dict[self.coalition_str(coalition)] = value
            return value

    def init_v(self, coalition, index=0):
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

    def generate_all_b(self, coalition, all_b, index=0):
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
dict_to_test = {"typ 1": [11, 12, 13], "typ 2": [21, 22, 23, 24], "typ 3": [31, 32, 33]}
cs = csg.create_coalition(dict_to_test)
print(cs)
"""
a = csg.get_q(list_to_test)
cs = csg.get_r(list_to_test)
print("optimal structure for {} has a value {} and is splitted as:".format(list_to_test, a))
output = ""
for coalition in cs:
    output += "({}, value: {}), ".format(coalition, csg.q_dict[csg.coalition_str(coalition)])
print(output[:-2])
"""
