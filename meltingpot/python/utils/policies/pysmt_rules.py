from pysmt.shortcuts import Not, LT, Int, is_sat

import numpy as np

class Rules():
    def __init__(self):
        self.rules = [Not(LT('num_apples_around', Int(2)))
                      ]

    def check(self, observation):
        """Returns True if a given rule holds, False if not"""
        for rule in self.rules:
            if not self.holds(rule, observation):
                return False
        return True
    
    def holds(self, rule, observation):
        """Returns if a rule holds given a certain observation."""
        properties = self.read_properties(rule)
        oberservations = self.get_properties(properties, observation)
        problem = rule.substitute([v for v in oberservations])
        return is_sat(problem)
    
    def read_properties(self, rule):
        """Returns a list of requested properties for rule satisfaction."""
        pass

    def get_properties(self, properties, observation):
        """Get the requested properties from the observations."""
        result = {}
        for item in properties:
            if not observation.item:
                return False
            
            result[item] = observation.item
        return result