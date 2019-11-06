from library import *

class ResourceManager():
    def __init__(self, minerals, gas, supply, bot: IDABot):
        self.resources = Resources(minerals, gas, supply)
        self.bot = bot

        # Reservations have an id as key, and resources used as value (Resources object)
        self.reservations = {}

        # Total reserved resources
        self.reserved = Resources(0, 0, 0)

    def sync(self):
        self.resources.minerals = self.bot.minerals
        self.resources.gas = self.bot.gas
        self.resources.supply = self.bot.current_supply

    def get_minerals(self):
        return self.resources.minerals - self.reserved.minerals

    def get_gas(self):
        return self.resources.gas - self.reserved.gas

    def get_supply(self):
        return self.resources.supply - self.reserved.supply

    def reserve(self, id, unit_type: UnitType):
        minerals = unit_type.mineral_price
        gas = unit_type.gas_price
        supply = unit_type.supply_required

        self.reserved.minerals += minerals
        self.reserved.gas += gas
        self.reserved.supply += supply

        self.reservations[id] = Resources(minerals, gas, supply)

    def release(self, id):
        self.reserved.minerals -= self.reservations[id].minerals
        self.reserved.gas -= self.reservations[id].gas
        self.reserved.supply -= self.reservations[id].supply
        del self.reservations[id]

    def use(self, unit_type: UnitType):
        self.resources.minerals -= unit_type.mineral_price
        self.resources.gas -= unit_type.gas_price
        self.resources.supply -= unit_type.supply_required

    def can_afford(self, unit_type: UnitType):
        """ Returns True if there are an sufficient amount of minerals, gas and supply to build the given unit_type,
        alse otherwise """
        return self.get_minerals >= unit_type.mineral_price and \
               self.get_gas >= unit_type.gas_price \
               and (self.bot.max_supply - self.get_supply) >= unit_type.supply_required

class Resources():
    def __init__(self, minerals, gas, supply):
        self.minerals = minerals
        self.gas = gas
        self.supply = supply

