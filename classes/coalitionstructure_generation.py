#from library import *
from typing import Dict, Any, List
from copy import deepcopy


class CoalitionstructureGeneration:
    """
    Gjort:
    Kom på sätt att spara v, q, r listor så att det är mycket snabbare, blir fett weird med nestade listor
    Fixa så att vi kan skapa ett max antal koalitioner

    Nästan gjorde:
    Integrera med boten, var ska funktionen kallas? ska csg klassen fördela specifika enheter? etc.

    Att göra (TODO):
    Effektivisera all_b delen? Kör all_b på start koalitionen och kolla på delmängder av den när vi kollar på mindre
        koalitioner?
    Värderings funktion!!
    """

    def __init__(self):
        # self.idabot = idabot
        self.v_list = None
        self.r_list = None
        self.q_list = None
        self.all_b = []
        self.nr_coal = 0
        return

    #    def create_coalition(self, military_units: Dict[UNIT_TYPEID: List[Unit.id]]) -> List[List[Unit.id]]:
    def create_coalition(self, military_units, nr_coalitions):
        """"
        Input should be a dictionary with UNIT_TYPEID as key and a list of unit ids of all units of that type as the
        value. Return is structured as [coalition1, coalition2, ...] where each coalition is a list as
        [[UnitId, unitId]], (UNIT_TYPEID, nr of units), ...]
        """

        self.nr_coal = nr_coalitions

        # Initialize type coalition with all units
        coalition = []
        for agent_count in military_units.values():
            coalition.append(len(agent_count))

        # Initialize lists v, q, r to nested lists with None values
        self.v_list = self.init_list(coalition)
        self.r_list = deepcopy(self.v_list)
        self.q_list = deepcopy(self.v_list)

        #Calculate optimal coalition structure with f and get it from r list
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

    def init_list(self, coalition, index = 0):
        """
        Create a nested list with structure coalition[0] lists which each contain coalition[1] lists which all contain
        coalition[2] lists ... to coalition[-1] which fills a list with coalition[-1] None values.
        :param coalition: The coalition to base list initilization on
        :param index: index in coalition
        :return: A nested list
        """
        if index == len(coalition):
            return None

        sublist = self.init_list(coalition, index+1)
        if sublist == None:
            return [None for i in range(coalition[index] + 1)]
        result = []
        for i in range(coalition[index] + 1):
            result.append(deepcopy(sublist))

        return result

    def f(self, coalition):
        """
        Calculate the optimal value of the input coalition. The coalition structure can be fetched with get_r function
        :param coalition: coalition to genereate optimal coalition structure for
        :return: optimal coalition structure value
        """
        # If the coalition is empty
        if (sum(coalition) == 0):
            return 0

        # Generate all possible list where every element <= corresponding coalition element
        # Remove first and last element as we dont want [0, 0, ..., 0] for neither set_of_b or new_col
        all_b = []
        self.generate_all_b(coalition, all_b)
        all_b = all_b[1:-1]

        max_value = self.get_v(coalition)
        max_coal = [coalition]
        for set_of_b in all_b:
            # Create new coalition [a1-b1, a2-b2, ...]
            new_col = []
            for i in range(len(set_of_b)):
                new_col.append(coalition[i] - set_of_b[i])

            # Calculate the value of this splitting of the coalition
            value = self.get_q(new_col) + self.get_v(set_of_b)

            # Check if the current cs has more coalitions than requested, if so continue to the next one
            if len(self.get_r(new_col)) >= self.nr_coal:
                continue

            if value > max_value:
                max_value = value
                max_coal = self.get_r(new_col) + [set_of_b]  # this is basically optimal structure for new_col + set of b

        self.set_r(coalition, max_coal)
        return max_value

    def get_q(self, coalition):
        """
        Get the optimal value for the coalition from Q table if it exists, calculate it otherwise.
        :param coalition: Coalition to get optiamal coalition structure value from
        :return: integer optimal coalition structure value
        """
        temp = self.get_list_element(self.q_list, coalition)
        if temp is None:
            f_value = self.f(coalition)
            self.set_list_element(self.q_list, coalition, f_value, "q_list")
            return f_value
        else:
            return temp

    def set_r(self, coalition, optimal_cs):
        self.set_list_element(self.r_list, coalition, optimal_cs, "r_list")

    def get_r(self, coalition):
        return self.get_list_element(self.r_list, coalition)

    def get_v(self, coalition):
        temp = self.get_list_element(self.v_list, coalition)
        if temp is None:
            value = self.v(coalition)
            self.set_list_element(self.v_list, coalition, value, "v_list")
            return value
        else:
            return temp

    def v(self, coalition: []) -> int:
        """
        Utility function, takes in a type coalition and returns a value for that type coalition
        :param coalition: coalition type to be evaluated
        :return: int
        """
        # TODO: Testa v med minsta koalition för olika antal av varje typ t.ex. [5, 3, 3]
        #  bästa borde vara [1, 1, 1], [2, 1, 1], [2, 1, 1]?
        value = 0
        for nr_type in coalition:
            c = 1
            for agent in range(nr_type):
                value += c
                c = c/2

        """
        if len(coalition) - coalition.count(0) != 1:
            return 0
        value = sum(map(lambda x: 1 + x ** 2, coalition))
        """
        """
        value = 0
        if coalition.count(0) > 0:
            return 0
        else:
            return 1 / sum(map(lambda x: x ** 2, coalition))
        #"""
        return value

    #TODO: kolla om detta är värt att göra?
    def init_v(self, coalition, index=0):
        """
        Calculate value of every possible coalition and save in v_dict.
        :param coalition: coalition type
        :return: None
        """
        if index >= len(coalition):
            #self.v_dict[self.coalition_str(coalition)] = self.v(coalition)
            return

        start_value = coalition[index]
        while coalition[index] >= 0:
            self.init_v(coalition, index + 1)
            coalition[index] -= 1
        # Restore the coalition to the same value as it started with
        coalition[index] = start_value

    def get_list_element(self, list, coalition):
        for elem in coalition:
            list = list[elem]
        return list

    def set_list_element(self, list, coalition, value, description = ""):
        for elem in coalition[:-1]:
            list = list[elem]
        list[coalition[-1]] = value

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


testing = False
if testing:
    csg = CoalitionstructureGeneration()
    b = []
    dict_to_test = {"typ 1": [11, 12, 13], "typ 2": [21, 22, 23, 24], "typ 3": [31, 32, 33],
                    "typ 4": [41, 42, 43, 44, 45, 46, 47, 48]}#, "typ 5": [51, 52, 53, 54]}#, "typ 6": [61, 62, 63, 64]}
    list_to_test = []
    for agent_count in dict_to_test.values():
        list_to_test.append(len(agent_count))
    cs = csg.create_coalition(dict_to_test, 3)
    print(cs)

    """
    a = csg.get_q(list_to_test)
    """
    cs = csg.get_r(list_to_test)
    print("total value: {}".format(csg.get_q(list_to_test)))
    output = ""
    for coalition in cs:
        output += "({}, value: {}), ".format(coalition, csg.get_q(coalition))
    print(output[:-2])
