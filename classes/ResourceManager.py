#TODO
#1. Test everything


class ResourceManager():
    def __init__(self, minerals, gas, supply, bot: IDAbot):
        # Resources are saved with id: [minerals, gas, supply]
        self.minerals = minerals
        self.gas = gas
        self.supply = supply
        self.bot = bot
        self.current_reservations = {}
        self.reserved = Reservation(0, 0, 0)

    def sync_resources(self):
        self.minerals = self.bot.minerals
        self.gas = self.bot.gas
        self.supply = self.bot.supply

    def get_minerals(self):
        return self.bot.minerals - self.reserved.minerals

    def get_gas(self):
        return self.bot.gas - self.reserved.gas

    def get_supply(self):
        return self.bot.supply - self.reserved.supply

    def reserve_resources(self, unit_type: UnitType, id):
        minerals_reserved = unit_type.mineral_price
        gas_reserved = unit_type.gas_price
        supply_reserved = unit_type.supply_required
        self.current_reservations[id] = Reservation(minerals_reserved, gas_reserved, supply_reserved)

    def release_reservation(self, id):
        del self.current_reservations[id]

    def use_resources(self, unit_type: UnitType):
        self.minerals -= unit_type.mineral_price
        self.gas -= unit_type.gas_price
        self.supply -= unit_type.supply

    def can_afford(self, unit_type: UnitType):
        """ Returns True if there are an sufficient amount of minerals, gas and supply to build the given unit_type,
        alse otherwise """
        return self.get_minerals >= unit_type.mineral_price and \
               self.get_gas >= unit_type.gas_price \
               and (bot.max_supply - self.get_supply) >= unit_type.supply_required

class Reservation():
    def __init__(self, minerals, gas, supply):
        self.minerals = minerals
        self.gas = gas
        self.supply = supply

