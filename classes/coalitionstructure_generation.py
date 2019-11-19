from library import *

class CoalitionstructureGeneration:

    self.MILITARY_TYPES = [UnitType(UNIT_TYPEID.TERRAN_MARINE, idabot),
                           UnitType(UNIT_TYPEID.TERRAN_MARAUDER, idabot),
                           UnitType(UNIT_TYPEID.TERRAN_REAPER, idabot), UnitType(UNIT_TYPEID.TERRAN_GHOST, idabot),
                           UnitType(UNIT_TYPEID.TERRAN_HELLION, idabot),
                           UnitType(UNIT_TYPEID.TERRAN_SIEGETANK, idabot),
                           UnitType(UNIT_TYPEID.TERRAN_CYCLONE, idabot),
                           UnitType(UNIT_TYPEID.TERRAN_WIDOWMINE, idabot),
                           UnitType(UNIT_TYPEID.TERRAN_THOR, idabot),
                           UnitType(UNIT_TYPEID.TERRAN_VIKINGASSAULT, idabot),
                           UnitType(UNIT_TYPEID.TERRAN_VIKINGFIGHTER, idabot),
                           UnitType(UNIT_TYPEID.TERRAN_MEDIVAC, idabot),
                           UnitType(UNIT_TYPEID.TERRAN_LIBERATOR, idabot),
                           UnitType(UNIT_TYPEID.TERRAN_RAVEN, idabot),
                           UnitType(UNIT_TYPEID.TERRAN_BANSHEE, idabot),
                           UnitType(UNIT_TYPEID.TERRAN_BATTLECRUISER, idabot),
                           UnitType(UNIT_TYPEID.TERRAN_AUTOTURRET, idabot)]


    def __init__(self, idabot):
        self.idabot = idabot
        return

    #Valuation function, takes in a coalition and returns a value for that coalition
    def v(self, coalitiontype : []) -> int:
        # TODO
        pass


    def create_coalition(self, type_coalition : {UNIT_TYPEID: int}) -> [[(UNIT_TYPEID, int)], [(UNIT_TYPEID, int)]]:
        """"
        Input should be a dictionary with UNIT_TYPEID as key and nr of units of that type as the value.
        Return is structured as [coalition1, coalition2, ...] where each coalition is a list as
        [(UNIT_TYPEID, nr of units), (UNIT_TYPEID, nr of units), ...]
        """
        coalition_structure = []
        return coalition_structure
