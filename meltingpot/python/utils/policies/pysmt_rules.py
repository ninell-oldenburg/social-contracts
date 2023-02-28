from pysmt.shortcuts import Not, LT, Int, is_sat

import numpy as np

class Rules():
    def __init__(self):
        self.rules = [Not(LT('num_apples_around', Int(2)))]

    def check(self, timestep, action):
        """Returns True if a given rule holds, False if not"""
        for rule in self.rules:
            if not self.holds(rule, timestep, action):
                return False
        return True
    
    def holds(self, rule, timestep, action):
        # unpack rule, substitute string
        properties = self.read_properties(rule)
        oberservations = self.get_properties(properties, timestep)
        problem = rule.substitute([v for v in oberservations])
        return is_sat(problem)