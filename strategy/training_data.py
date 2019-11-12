class MatchState:
    def __init__(self):
        # Observed game state
        self.resources = None               # {low, mid, high}
        self.expansions = None              # {0, ..., 16}
        self.armies = None                  # [0,1]
        self.defensive_buildings = None     # [0,1]
        self.production_buildings = None    # [0,1]

        self.match_stage = None             # {early, mid, late}

        self.observed_armies = None         # [0,1]
        self.observed_defensive = None      # [0,1]
        self.observed_strategy = None       # [0,1] (None = 0.25 , Offensive: 0.50, Defensive: 0.75, Expansive: 1}

        # Classified from state
        self.classified_strategy = None     # {Offensive, Defensive or Expansive}

    def get_short_parameters_1(self):
        return [self.armies, self.match_stage]

    def get_parameters(self):
        return [self.resources, self.expansions, self.armies, self.defensive_buildings, self.production_buildings,
                self.match_stage, self.observed_armies, self.observed_defensive, self.observed_strategy]

    def get_output(self):
        return self.classified_strategy